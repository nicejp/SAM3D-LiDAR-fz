import SwiftUI
import ARKit
import RealityKit

struct ContentView: View {
    @StateObject private var arManager = ARManager()
    @State private var isScanning = false
    @State private var serverIP = "10.255.255.225"
    @State private var statusMessage = "待機中"

    var body: some View {
        ZStack {
            ARViewContainer(arManager: arManager)
                .edgesIgnoringSafeArea(.all)

            VStack {
                // ステータス表示
                VStack(spacing: 5) {
                    Text(statusMessage)
                        .font(.headline)
                        .foregroundColor(.white)
                    Text("頂点数: \(arManager.vertexCount)")
                        .font(.subheadline)
                        .foregroundColor(.cyan)
                    Text("RGB: \(arManager.rgbFrameCount) フレーム")
                        .font(.subheadline)
                        .foregroundColor(.green)
                }
                .padding()
                .background(Color.black.opacity(0.7))
                .cornerRadius(10)
                .padding(.top, 50)

                Spacer()

                // サーバーIP入力
                HStack {
                    Text("Server:")
                        .foregroundColor(.white)
                    TextField("IP Address", text: $serverIP)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 150)
                        .keyboardType(.decimalPad)
                }
                .padding()
                .background(Color.black.opacity(0.7))
                .cornerRadius(10)

                // スキャンボタン
                Button(action: {
                    isScanning.toggle()
                    if isScanning {
                        arManager.startCapture(serverIP: serverIP)
                        statusMessage = "スキャン中..."
                    } else {
                        arManager.stopCapture()
                        statusMessage = "停止"
                    }
                }) {
                    HStack {
                        Image(systemName: isScanning ? "stop.circle.fill" : "record.circle")
                            .font(.title)
                        Text(isScanning ? "停止" : "スキャン開始")
                            .font(.headline)
                    }
                    .foregroundColor(.white)
                    .padding()
                    .frame(width: 200)
                    .background(isScanning ? Color.red : Color.blue)
                    .cornerRadius(15)
                }
                .padding(.bottom, 50)
            }
        }
        .onReceive(arManager.$frameCount) { count in
            if isScanning {
                statusMessage = "スキャン中: \(count) フレーム"
            }
        }
        .onReceive(arManager.$connectionStatus) { status in
            if !isScanning {
                statusMessage = status
            }
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    var arManager: ARManager

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arManager.setupARView(arView)
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}

#Preview {
    ContentView()
}
