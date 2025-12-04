import ARKit
import RealityKit
import Combine

/// ARセッションを管理し、LiDAR + RGBデータを取得・送信するクラス
class ARManager: NSObject, ObservableObject {
    private var arView: ARView?
    private var webSocket: URLSessionWebSocketTask?
    private var isCapturing = false

    @Published var frameCount = 0
    @Published var connectionStatus = "未接続"
    @Published var vertexCount = 0
    @Published var rgbFrameCount = 0

    // メッシュデータ
    private var meshAnchors: [UUID: ARMeshAnchor] = [:]
    private var allVertices: [SIMD3<Float>] = []

    // RGB/深度送信用のカウンター
    private var internalFrameCount = 0

    // MARK: - Setup

    func setupARView(_ arView: ARView) {
        self.arView = arView

        // LiDAR対応の設定（メッシュ再構築を有効化）
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]

        // LiDARメッシュ再構築を有効化
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
            print("LiDARメッシュ再構築を有効化")
        }

        // LiDAR確認
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            connectionStatus = "エラー: LiDAR非対応デバイス"
            print("このデバイスはLiDARに対応していません")
            return
        }

        arView.session.delegate = self
        arView.session.run(config)
        connectionStatus = "待機中"
    }

    // MARK: - Capture Control

    func startCapture(serverIP: String) {
        // WebSocket接続
        guard let url = URL(string: "ws://\(serverIP):8765") else {
            connectionStatus = "エラー: 無効なIPアドレス"
            return
        }

        let session = URLSession(configuration: .default)
        webSocket = session.webSocketTask(with: url)
        webSocket?.resume()

        isCapturing = true
        frameCount = 0
        rgbFrameCount = 0
        internalFrameCount = 0
        connectionStatus = "接続中..."

        // 接続確認
        receiveMessage()

        print("キャプチャ開始: \(serverIP)")
    }

    func stopCapture() {
        isCapturing = false
        webSocket?.cancel(with: .goingAway, reason: nil)
        connectionStatus = "停止: \(frameCount) フレーム送信済み"
        print("キャプチャ停止: \(frameCount) フレーム")
    }

    private func receiveMessage() {
        webSocket?.receive { [weak self] result in
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    print("サーバーからの応答: \(text)")
                    DispatchQueue.main.async {
                        self?.connectionStatus = "接続済み"
                    }
                default:
                    break
                }
                self?.receiveMessage()
            case .failure(let error):
                print("受信エラー: \(error)")
                DispatchQueue.main.async {
                    self?.connectionStatus = "接続エラー"
                }
            }
        }
    }
}

// MARK: - ARSessionDelegate

extension ARManager: ARSessionDelegate {

    // メッシュアンカーが追加された時
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                meshAnchors[meshAnchor.identifier] = meshAnchor
                updateAllVertices()
            }
        }
    }

    // メッシュアンカーが更新された時
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                meshAnchors[meshAnchor.identifier] = meshAnchor
                updateAllVertices()
            }
        }
    }

    // メッシュアンカーが削除された時
    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors {
            if let meshAnchor = anchor as? ARMeshAnchor {
                meshAnchors.removeValue(forKey: meshAnchor.identifier)
                updateAllVertices()
            }
        }
    }

    // 全頂点を更新
    private func updateAllVertices() {
        var vertices: [SIMD3<Float>] = []

        for (_, anchor) in meshAnchors {
            let geometry = anchor.geometry
            let transform = anchor.transform

            let vertexBuffer = geometry.vertices
            let vertexCount = geometry.vertices.count
            let vertexStride = geometry.vertices.stride

            for i in 0..<vertexCount {
                let vertexPointer = vertexBuffer.buffer.contents().advanced(by: vertexBuffer.offset + i * vertexStride)
                let vertex = vertexPointer.assumingMemoryBound(to: SIMD3<Float>.self).pointee

                // ワールド座標に変換
                let worldPosition = transform * SIMD4<Float>(vertex.x, vertex.y, vertex.z, 1.0)
                let position = SIMD3<Float>(worldPosition.x, worldPosition.y, worldPosition.z)

                vertices.append(position)
            }
        }

        allVertices = vertices
        DispatchQueue.main.async {
            self.vertexCount = vertices.count
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isCapturing else { return }

        internalFrameCount += 1

        // カメラパラメータ
        let camera = frame.camera
        let transform = camera.transform
        let cameraPosition = SIMD3<Float>(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)

        // 3fpsでメッシュ頂点を送信（毎10フレーム = 30fps / 10 = 3fps）
        if internalFrameCount % 10 == 0 {
            frameCount += 1

            // メッシュ頂点をサンプリング（最大30000点 - WiFi負荷軽減）
            var verticesToSend: [[Float]] = []
            let maxVertices = 30000

            if allVertices.count > 0 {
                let step = max(1, allVertices.count / maxVertices)
                for i in stride(from: 0, to: allVertices.count, by: step) {
                    let v = allVertices[i]
                    verticesToSend.append([v.x, v.y, v.z])
                }
            }

            // メッシュ頂点データをJSON形式でパッケージ
            var packet: [String: Any] = [
                "type": "mesh_vertices",
                "frame_id": frameCount,
                "timestamp": frame.timestamp,
                "vertices": verticesToSend,
                "vertex_count": verticesToSend.count,
                "total_vertices": allVertices.count,
                "camera_position": [cameraPosition.x, cameraPosition.y, cameraPosition.z],
                "transform": transformToArray(transform)
            ]

            // WebSocketで送信
            if let jsonData = try? JSONSerialization.data(withJSONObject: packet),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                webSocket?.send(.string(jsonString)) { [weak self] error in
                    if let error = error {
                        print("送信エラー: \(error)")
                        DispatchQueue.main.async {
                            self?.connectionStatus = "送信エラー"
                        }
                    }
                }
            }
        }

        // 1fpsでRGB画像と深度マップを送信（毎30フレーム = 30fps / 30 = 1fps）
        if internalFrameCount % 30 == 0 {
            rgbFrameCount += 1

            var rgbPacket: [String: Any] = [
                "type": "frame_data",
                "frame_id": rgbFrameCount,
                "timestamp": frame.timestamp,
                "camera_position": [cameraPosition.x, cameraPosition.y, cameraPosition.z],
                "transform": transformToArray(transform)
            ]

            // RGB画像を送信
            if let capturedImage = frame.capturedImage,
               let jpegData = pixelBufferToJPEG(capturedImage) {
                rgbPacket["rgb"] = jpegData.base64EncodedString()
            }

            // 深度マップを送信
            if let sceneDepth = frame.sceneDepth {
                let depthMap = sceneDepth.depthMap
                let depthData = depthMapToData(depthMap)
                rgbPacket["depth"] = depthData.base64EncodedString()
                rgbPacket["depth_width"] = CVPixelBufferGetWidth(depthMap)
                rgbPacket["depth_height"] = CVPixelBufferGetHeight(depthMap)
            }

            // カメラ内部パラメータを送信
            let intrinsics = camera.intrinsics
            rgbPacket["intrinsics"] = [
                "fx": intrinsics[0][0],
                "fy": intrinsics[1][1],
                "cx": intrinsics[2][0],
                "cy": intrinsics[2][1]
            ]

            // WebSocketで送信
            if let jsonData = try? JSONSerialization.data(withJSONObject: rgbPacket),
               let jsonString = String(data: jsonData, encoding: .utf8) {
                webSocket?.send(.string(jsonString)) { [weak self] error in
                    if let error = error {
                        print("RGB送信エラー: \(error)")
                    } else {
                        print("RGB フレーム \(self?.rgbFrameCount ?? 0) 送信完了")
                    }
                }
            }
        }
    }
}

// MARK: - Helper Functions

extension ARManager {
    private func pixelBufferToJPEG(_ pixelBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }
        let uiImage = UIImage(cgImage: cgImage)
        return uiImage.jpegData(compressionQuality: 0.8)
    }

    private func depthMapToData(_ depthMap: CVPixelBuffer) -> Data {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        let baseAddress = CVPixelBufferGetBaseAddress(depthMap)!

        let floatBuffer = baseAddress.assumingMemoryBound(to: Float32.self)
        let count = width * height
        let data = Data(bytes: floatBuffer, count: count * MemoryLayout<Float32>.size)

        return data
    }

    private func transformToArray(_ transform: simd_float4x4) -> [Float] {
        return [
            transform.columns.0.x, transform.columns.0.y, transform.columns.0.z, transform.columns.0.w,
            transform.columns.1.x, transform.columns.1.y, transform.columns.1.z, transform.columns.1.w,
            transform.columns.2.x, transform.columns.2.y, transform.columns.2.z, transform.columns.2.w,
            transform.columns.3.x, transform.columns.3.y, transform.columns.3.z, transform.columns.3.w
        ]
    }
}
