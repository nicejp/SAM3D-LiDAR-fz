#!/usr/bin/env python3
"""
Web-based Realtime Point Cloud Viewer
iPadからのデータをリアルタイムで3D表示するWebビューワー

Three.jsを使用してブラウザで表示
iPadアプリのMiniMap3DViewと同等の機能を提供

使い方:
    python -m server.visualization.web_viewer --port 8765
    ブラウザで http://localhost:8080 を開く
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import io
import base64
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser

try:
    import websockets
    from websockets.server import serve
    from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Error: websockets is required. Run: pip install websockets")

from PIL import Image


# HTML/JavaScript for the web viewer
WEB_VIEWER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Realtime 3D Mesh Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; background: #1a1a2e; font-family: sans-serif; }
        #container { width: 100vw; height: 100vh; }
        #info {
            position: absolute; top: 10px; left: 10px;
            background: rgba(0,0,0,0.8); color: #fff;
            padding: 15px; border-radius: 8px;
            font-size: 14px; z-index: 100;
            min-width: 200px;
        }
        #info h3 { margin: 0 0 10px 0; color: #0ff; }
        #info div { margin: 5px 0; }
        .stat-label { color: #888; }
        .stat-value { color: #0f0; font-weight: bold; }
        #legend {
            position: absolute; bottom: 10px; left: 10px;
            background: rgba(0,0,0,0.8); color: #fff;
            padding: 10px; border-radius: 8px;
            font-size: 12px; z-index: 100;
        }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        #status {
            position: absolute; top: 10px; right: 10px;
            padding: 10px 15px; border-radius: 5px;
            font-size: 14px; z-index: 100;
        }
        .connected { background: #0a0; color: #fff; }
        .disconnected { background: #a00; color: #fff; }
        .connecting { background: #aa0; color: #000; }
        #controls {
            position: absolute; bottom: 10px; right: 10px;
            background: rgba(0,0,0,0.8); color: #fff;
            padding: 10px; border-radius: 8px;
            font-size: 12px; z-index: 100;
        }
        button {
            background: #333; color: #fff; border: 1px solid #555;
            padding: 8px 15px; margin: 3px; border-radius: 5px; cursor: pointer;
        }
        button:hover { background: #555; }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>3D Mesh Viewer</h3>
        <div><span class="stat-label">Mesh Points:</span> <span class="stat-value" id="pointCount">0</span></div>
        <div><span class="stat-label">Total Vertices:</span> <span class="stat-value" id="totalVertices">0</span></div>
        <div><span class="stat-label">Frames:</span> <span class="stat-value" id="frameCount">0</span></div>
        <div><span class="stat-label">Data Rate:</span> <span class="stat-value" id="dataRate">0</span> <span class="stat-label">fps</span></div>
    </div>
    <div id="legend">
        <div class="legend-item"><div class="legend-color" style="background:#0f0;"></div>Camera Position</div>
        <div class="legend-item"><div class="legend-color" style="background:#f00;"></div>Camera Trail</div>
        <div class="legend-item"><div class="legend-color" style="background:linear-gradient(#f00,#0f0);"></div>Mesh (height)</div>
        <div style="margin-top: 10px; color: #888;">Drag to rotate / Scroll to zoom</div>
    </div>
    <div id="status" class="disconnected">Disconnected</div>
    <div id="controls">
        <button onclick="resetView()">Reset View</button>
        <button onclick="clearPoints()">Clear</button>
        <button onclick="toggleAutoRotate()">Auto Rotate</button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);

        // Add fog for depth effect
        scene.fog = new THREE.Fog(0x1a1a2e, 10, 50);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(3, 3, 3);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = false;
        controls.autoRotateSpeed = 1.0;

        // Grid (XZ plane)
        const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
        scene.add(gridHelper);

        // Axes helper
        const axesHelper = new THREE.AxesHelper(1);
        scene.add(axesHelper);

        // Point cloud for mesh vertices
        let meshGeometry = new THREE.BufferGeometry();
        let meshMaterial = new THREE.PointsMaterial({
            size: 0.015,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: true,
            opacity: 0.9
        });
        let meshPoints = new THREE.Points(meshGeometry, meshMaterial);
        scene.add(meshPoints);

        // Camera marker (cone pointing forward)
        const cameraGroup = new THREE.Group();
        const cameraMarkerGeometry = new THREE.ConeGeometry(0.08, 0.2, 8);
        const cameraMarkerMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const cameraMarker = new THREE.Mesh(cameraMarkerGeometry, cameraMarkerMaterial);
        cameraMarker.rotation.x = -Math.PI / 2;  // Point forward (-Z)
        cameraGroup.add(cameraMarker);

        // Camera wireframe box
        const cameraBoxGeometry = new THREE.BoxGeometry(0.15, 0.1, 0.05);
        const cameraBoxMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
        const cameraBox = new THREE.Mesh(cameraBoxGeometry, cameraBoxMaterial);
        cameraBox.position.z = 0.1;
        cameraGroup.add(cameraBox);
        scene.add(cameraGroup);

        // Camera trail
        const trailPoints = [];
        const maxTrailPoints = 500;
        let trailGeometry = new THREE.BufferGeometry();
        const trailMaterial = new THREE.LineBasicMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.7
        });
        const trailLine = new THREE.Line(trailGeometry, trailMaterial);
        scene.add(trailLine);

        // State
        let frameCount = 0;
        let totalVertices = 0;
        let dataRateCounter = 0;  // Data frames received per second
        let ws = null;

        // Helper functions
        function updateStatus(status, className) {
            const el = document.getElementById('status');
            el.textContent = status;
            el.className = className;
        }

        function resetView() {
            camera.position.set(3, 3, 3);
            camera.lookAt(0, 0, 0);
            controls.reset();
        }

        function clearPoints() {
            meshGeometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
            meshGeometry.setAttribute('color', new THREE.Float32BufferAttribute([], 3));
            trailPoints.length = 0;
            trailGeometry.setFromPoints([]);
            frameCount = 0;
            totalVertices = 0;
            document.getElementById('pointCount').textContent = '0';
            document.getElementById('totalVertices').textContent = '0';
            document.getElementById('frameCount').textContent = '0';
        }

        function toggleAutoRotate() {
            controls.autoRotate = !controls.autoRotate;
        }

        // WebSocket connection
        function connectWebSocket() {
            updateStatus('Connecting...', 'connecting');
            ws = new WebSocket(`ws://${window.location.hostname}:WS_PORT_PLACEHOLDER`);

            ws.onopen = () => {
                updateStatus('Connected', 'connected');
                console.log('WebSocket connected');
            };

            ws.onclose = () => {
                updateStatus('Disconnected', 'disconnected');
                setTimeout(connectWebSocket, 2000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                } catch (e) {
                    console.error('Parse error:', e);
                }
            };
        }

        function handleMessage(data) {
            // Handle mesh vertices (full replacement each frame)
            if (data.type === 'points' || data.type === 'mesh') {
                const vertices = data.points;
                if (!vertices || vertices.length === 0) return;

                // Build position and color arrays
                const positions = new Float32Array(vertices.length * 3);
                const colors = new Float32Array(vertices.length * 3);

                // Find Y range for color mapping
                let yMin = Infinity, yMax = -Infinity;
                for (let i = 0; i < vertices.length; i++) {
                    const y = vertices[i][1];
                    if (y < yMin) yMin = y;
                    if (y > yMax) yMax = y;
                }
                const yRange = Math.max(yMax - yMin, 0.1);

                // Fill arrays
                for (let i = 0; i < vertices.length; i++) {
                    positions[i * 3] = vertices[i][0];
                    positions[i * 3 + 1] = vertices[i][1];
                    positions[i * 3 + 2] = vertices[i][2];

                    // Color by height (blue at bottom, green at top)
                    const t = (vertices[i][1] - yMin) / yRange;
                    colors[i * 3] = 0.2;           // R
                    colors[i * 3 + 1] = 0.3 + t * 0.7;  // G
                    colors[i * 3 + 2] = 1.0 - t * 0.5;  // B
                }

                // Update geometry (REPLACE, not accumulate)
                meshGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                meshGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                meshGeometry.computeBoundingSphere();

                frameCount++;
                dataRateCounter++;  // Count data updates per second
                totalVertices = data.total_vertices || vertices.length;

                document.getElementById('pointCount').textContent = vertices.length.toLocaleString();
                document.getElementById('totalVertices').textContent = totalVertices.toLocaleString();
                document.getElementById('frameCount').textContent = frameCount;
            }

            // Handle camera pose
            if (data.type === 'camera_pose' || data.camera_position) {
                const pos = data.camera_position;
                if (pos) {
                    cameraGroup.position.set(pos[0], pos[1], pos[2]);

                    // カメラの回転を適用（transformから）
                    if (data.transform && data.transform.length === 16) {
                        const t = data.transform;
                        // ARKit: column-major 4x4 matrix
                        // columns: [0-3], [4-7], [8-11], [12-15]
                        // Forward direction is -Z axis (column 2, negated)
                        const forward = new THREE.Vector3(-t[8], -t[9], -t[10]).normalize();
                        const up = new THREE.Vector3(t[4], t[5], t[6]).normalize();

                        // カメラマーカーの向きを設定
                        const lookAtPoint = new THREE.Vector3(
                            pos[0] + forward.x,
                            pos[1] + forward.y,
                            pos[2] + forward.z
                        );
                        cameraGroup.lookAt(lookAtPoint);
                    }

                    // Add to trail
                    trailPoints.push(new THREE.Vector3(pos[0], pos[1], pos[2]));
                    if (trailPoints.length > maxTrailPoints) {
                        trailPoints.shift();
                    }
                    trailGeometry.setFromPoints(trailPoints);
                }
            }

            // Handle reset command
            if (data.type === 'reset') {
                clearPoints();
            }
        }

        // Data rate counter
        setInterval(() => {
            document.getElementById('dataRate').textContent = dataRateCounter;
            dataRateCounter = 0;
        }, 1000);

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Start
        connectWebSocket();
        animate();
    </script>
</body>
</html>
"""


class WebPointCloudViewer:
    """Web版リアルタイム点群ビューワー"""

    def __init__(self, ws_port: int = 8765, http_port: int = 8080, save_dir: str = "~/LiDAR-LLM-MCP/experiments"):
        self.ws_port = ws_port
        self.http_port = http_port
        self.save_dir = Path(save_dir).expanduser()

        # 接続中のWebクライアント
        self.web_clients = set()

        # セッション管理
        self.session_dir = None
        self.frame_count = 0
        self.total_points = 0

        # 制御
        self.running = True

    def start_session(self) -> Path:
        """新しいセッションを開始"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.save_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "rgb").mkdir(exist_ok=True)
        (self.session_dir / "depth").mkdir(exist_ok=True)
        (self.session_dir / "camera").mkdir(exist_ok=True)
        self.frame_count = 0
        print(f"[Session] Started: {self.session_dir}")
        return self.session_dir

    def save_frame(self, frame_id: int, rgb_bytes: bytes, depth_array: np.ndarray, camera_data: dict):
        """フレームデータを保存"""
        if self.session_dir is None:
            return

        try:
            if rgb_bytes:
                rgb_path = self.session_dir / "rgb" / f"frame_{frame_id:06d}.jpg"
                with open(rgb_path, "wb") as f:
                    f.write(rgb_bytes)

            if depth_array is not None:
                depth_path = self.session_dir / "depth" / f"frame_{frame_id:06d}.npy"
                np.save(depth_path, depth_array)

            camera_path = self.session_dir / "camera" / f"frame_{frame_id:06d}.json"
            with open(camera_path, "w") as f:
                json.dump(camera_data, f, indent=2)

        except Exception as e:
            print(f"[Error] Save failed: {e}")

    def depth_to_points(
        self,
        depth_data: np.ndarray,
        intrinsics: dict,
        rgb_data: Optional[np.ndarray] = None,
        transform: Optional[list] = None
    ) -> tuple:
        """深度マップから点群を生成（ARKit座標系）"""
        height, width = depth_data.shape

        fx = intrinsics.get("fx", 500.0)
        fy = intrinsics.get("fy", 500.0)
        cx = intrinsics.get("cx", width / 2)
        cy = intrinsics.get("cy", height / 2)

        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)

        z = depth_data
        valid = (z > 0.1) & (z < 5.0) & np.isfinite(z)

        # ARKitカメラ座標系に変換
        x = (u - cx) * z / fx
        y = (cy - v) * z / fy  # Y軸反転（ARKitは上が正）
        z_camera = -z          # Z軸反転（ARKitは-Zが前方）

        points = np.stack([x[valid], y[valid], z_camera[valid]], axis=-1)

        # ワールド座標系に変換
        if transform is not None and len(transform) == 16:
            transform_matrix = np.array(transform).reshape(4, 4).T
            ones = np.ones((points.shape[0], 1))
            points_homo = np.hstack([points, ones])
            points = (transform_matrix @ points_homo.T).T[:, :3]

        colors = None
        if rgb_data is not None:
            if rgb_data.shape[:2] != depth_data.shape:
                rgb_pil = Image.fromarray(rgb_data)
                rgb_pil = rgb_pil.resize((width, height), Image.BILINEAR)
                rgb_data = np.array(rgb_pil)
            colors = rgb_data[valid] / 255.0

        return points, colors

    async def broadcast_to_web(self, message: dict):
        """Webクライアントにブロードキャスト"""
        if self.web_clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_str) for client in self.web_clients],
                return_exceptions=True
            )

    async def handle_ipad_client(self, websocket):
        """iPadからのWebSocket接続を処理"""
        client_addr = websocket.remote_address
        print(f"[iPad] Connected: {client_addr}")

        self.start_session()
        await websocket.send(json.dumps({"status": "connected", "message": "Web Viewer Ready"}))

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_frame(data, websocket)
                except json.JSONDecodeError as e:
                    print(f"[Error] JSON parse error: {e}")

        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            pass  # 正常な切断
        except Exception as e:
            print(f"[Error] {e}")
        finally:
            if self.session_dir:
                metadata = {
                    "total_frames": self.frame_count,
                    "total_points": self.total_points
                }
                metadata_path = self.session_dir / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                print(f"[Session] Ended: {self.frame_count} frames")
            print(f"[iPad] Disconnected: {client_addr}")

    async def handle_web_client(self, websocket):
        """Webブラウザからの接続を処理"""
        print(f"[Web] Client connected")
        self.web_clients.add(websocket)
        try:
            async for message in websocket:
                # Webクライアントからのコマンド処理（将来の拡張用）
                pass
        except Exception as e:
            pass
        finally:
            self.web_clients.discard(websocket)
            print(f"[Web] Client disconnected")

    async def process_frame(self, data: dict, websocket):
        """フレームデータを処理してWebクライアントに転送"""
        try:
            frame_id = data.get("frame_id", self.frame_count)
            msg_type = data.get("type", "depth")

            # メッシュ頂点形式（新形式）- ARKitから直接頂点を受信
            if msg_type == "mesh_vertices":
                vertices = data.get("vertices", [])
                camera_pos = data.get("camera_position", [0, 0, 0])
                transform = data.get("transform", None)

                if vertices:
                    # Webクライアントに直接転送（座標変換不要）
                    web_message = {
                        "type": "points",
                        "points": vertices,
                        "colors": None  # 高さベースで色付け（クライアント側）
                    }
                    await self.broadcast_to_web(web_message)

                    # カメラ位置と向きを送信
                    await self.broadcast_to_web({
                        "type": "camera_pose",
                        "camera_position": camera_pos,
                        "transform": transform  # カメラの向き用
                    })

                    self.frame_count += 1
                    self.total_points = data.get("total_vertices", len(vertices))

                    # メッシュ頂点をファイルに保存（10フレームごと + 最新を常に保存）
                    if self.session_dir:
                        mesh_dir = self.session_dir / "mesh"
                        mesh_dir.mkdir(exist_ok=True)

                        # NumPy形式で保存
                        vertices_array = np.array(vertices, dtype=np.float32)

                        # 最新のメッシュを常に保存（latest.npy）
                        np.save(mesh_dir / "latest.npy", vertices_array)

                        # 10フレームごとにスナップショット保存
                        if self.frame_count % 10 == 0:
                            np.save(mesh_dir / f"frame_{frame_id:06d}.npy", vertices_array)
                            print(f"[Frame {self.frame_count}] Mesh: {len(vertices)} pts (total: {self.total_points})")

                # カメラ位置を保存
                camera_data = {
                    "frame_id": frame_id,
                    "timestamp": data.get("timestamp", 0),
                    "camera_position": camera_pos,
                    "total_vertices": self.total_points
                }
                if self.session_dir:
                    camera_path = self.session_dir / "camera" / f"frame_{frame_id:06d}.json"
                    with open(camera_path, "w") as f:
                        json.dump(camera_data, f, indent=2)

                try:
                    await websocket.send(json.dumps({"status": "ok", "frame_id": frame_id}))
                except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
                    pass  # 切断時は無視
                return

            # 深度マップ形式（旧形式）
            # RGB画像をデコード
            rgb_image = None
            rgb_bytes_raw = None
            if "rgb" in data and data["rgb"]:
                rgb_bytes_raw = base64.b64decode(data["rgb"])
                rgb_image = np.array(Image.open(io.BytesIO(rgb_bytes_raw)))

            # 深度マップをデコード
            depth_map = None
            if "depth" in data and data["depth"]:
                depth_bytes = base64.b64decode(data["depth"])
                width = data.get("depth_width", 256)
                height = data.get("depth_height", 192)
                depth_map = np.frombuffer(depth_bytes, dtype=np.float32).reshape((height, width))

            # カメラパラメータ
            intrinsics = data.get("intrinsics", {})
            transform = data.get("transform", None)

            # データを保存
            camera_data = {
                "frame_id": frame_id,
                "timestamp": data.get("timestamp", 0),
                "intrinsics": intrinsics,
                "transform": transform if transform else []
            }
            self.save_frame(frame_id, rgb_bytes_raw, depth_map, camera_data)

            if depth_map is not None:
                # 点群に変換
                points, colors = self.depth_to_points(depth_map, intrinsics, rgb_image, transform)

                if len(points) > 0:
                    # サブサンプリング（転送用）
                    if len(points) > 5000:
                        indices = np.random.choice(len(points), 5000, replace=False)
                        points_send = points[indices]
                        colors_send = colors[indices] if colors is not None else None
                    else:
                        points_send = points
                        colors_send = colors

                    # Webクライアントに送信
                    web_message = {
                        "type": "points",
                        "points": points_send.tolist(),
                        "colors": colors_send.tolist() if colors_send is not None else None
                    }
                    await self.broadcast_to_web(web_message)

                    # カメラ位置も送信
                    if transform and len(transform) == 16:
                        transform_matrix = np.array(transform).reshape(4, 4).T
                        camera_pos = transform_matrix[:3, 3].tolist()
                        await self.broadcast_to_web({
                            "type": "camera_pose",
                            "camera_position": camera_pos
                        })

                    self.frame_count += 1
                    self.total_points += len(points_send)

                    if self.frame_count % 10 == 0:
                        print(f"[Frame {self.frame_count}] +{len(points_send)} pts, Total: {self.total_points}")

            # 確認応答
            try:
                await websocket.send(json.dumps({"status": "ok", "frame_id": frame_id}))
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
                pass  # 切断時は無視

        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            # 接続切断は正常動作なのでエラー表示しない
            pass
        except Exception as e:
            print(f"[Error] Frame processing: {e}")

    async def run_ipad_server(self):
        """iPad用WebSocketサーバー"""
        print(f"[WS] iPad server on port {self.ws_port}")
        async with serve(self.handle_ipad_client, "0.0.0.0", self.ws_port, max_size=50*1024*1024):
            while self.running:
                await asyncio.sleep(0.1)

    async def run_web_server(self):
        """Webクライアント用WebSocketサーバー"""
        web_ws_port = self.ws_port + 1
        print(f"[WS] Web client server on port {web_ws_port}")
        async with serve(self.handle_web_client, "0.0.0.0", web_ws_port):
            while self.running:
                await asyncio.sleep(0.1)

    def run_http_server(self):
        """HTTPサーバー（HTMLを配信）"""
        web_ws_port = self.ws_port + 1
        html_content = WEB_VIEWER_HTML.replace("WS_PORT_PLACEHOLDER", str(web_ws_port))

        class Handler(SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(html_content.encode())
                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass  # 静かに

        server = HTTPServer(("0.0.0.0", self.http_port), Handler)
        print(f"[HTTP] Server on http://0.0.0.0:{self.http_port}")
        server.serve_forever()

    async def run_async(self):
        """非同期メイン"""
        await asyncio.gather(
            self.run_ipad_server(),
            self.run_web_server()
        )

    def run(self):
        """メイン実行"""
        if not HAS_WEBSOCKETS:
            print("Error: pip install websockets")
            return

        print("\n" + "=" * 60)
        print("Web-based Realtime Point Cloud Viewer")
        print("=" * 60)
        print(f"iPad WebSocket:  ws://0.0.0.0:{self.ws_port}")
        print(f"Web Viewer:      http://0.0.0.0:{self.http_port}")
        print(f"Data save:       {self.save_dir}")
        print("=" * 60)
        print("1. Open the web viewer in browser")
        print("2. Connect iPad app to the WebSocket port")
        print("3. Start scanning!")
        print("=" * 60 + "\n")

        # HTTPサーバーをバックグラウンドで起動
        http_thread = threading.Thread(target=self.run_http_server, daemon=True)
        http_thread.start()

        # ブラウザを自動で開く
        import time
        time.sleep(0.5)  # サーバー起動を待つ
        url = f"http://localhost:{self.http_port}"
        print(f"Opening browser: {url}")
        webbrowser.open(url)

        # 非同期サーバーを実行
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Web-based Realtime Point Cloud Viewer")
    parser.add_argument("--port", "-p", type=int, default=8765, help="WebSocket port for iPad")
    parser.add_argument("--http-port", type=int, default=8080, help="HTTP port for web viewer")
    args = parser.parse_args()

    viewer = WebPointCloudViewer(ws_port=args.port, http_port=args.http_port)
    viewer.run()


if __name__ == "__main__":
    main()
