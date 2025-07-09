class Navball {
	constructor(containerId, size = null) {
		this.container = document.getElementById(containerId);
		this.size = size || CONFIG.navball.size;

		this.yaw = 0;
		this.pitch = 0;
		this.isDragging = false;
		this.lastMouseX = 0;
		this.lastMouseY = 0;
		this.sensitivity = CONFIG.navball.sensitivity;

		this.init();
	}

	init() {
		this.setupThreeJS();
		this.createNavball();
		this.setupControls();
		this.animate();
	}

	setupThreeJS() {
		this.scene = new THREE.Scene();
		this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
		this.camera.position.z = 3;

		this.renderer = new THREE.WebGLRenderer({
			antialias: true,
			alpha: true
		});
		this.renderer.setSize(this.size, this.size);
		this.renderer.setClearColor(0x000000, 0);
		this.renderer.domElement.id = 'navball-canvas';
		this.container.appendChild(this.renderer.domElement);
	}

	createNavball() {
		// Low-density wireframe sphere - using CONFIG values
		const geometry = new THREE.SphereGeometry(
			1,
			CONFIG.navball.sphereDetail.widthSegments,
			CONFIG.navball.sphereDetail.heightSegments
		);
		const material = new THREE.MeshBasicMaterial({
			color: 0x004400,
			wireframe: true,
			transparent: true,
			opacity: 0.3
		});

		this.navballMesh = new THREE.Mesh(geometry, material);
		this.scene.add(this.navballMesh);

		// Axis group
		this.axisGroup = new THREE.Group();
		this.scene.add(this.axisGroup);

		this.createAxes();
	}

	createAxes() {
		const axisLength = CONFIG.navball.axisLength;
		const arrowLength = CONFIG.navball.arrowLength;
		const arrowRadius = CONFIG.navball.arrowRadius;

		const axes = [
			{ name: 'X', color: 0xff4444, direction: [1, 0, 0], rotation: [0, 0, -Math.PI / 2] },
			{ name: 'Y', color: 0x44ff44, direction: [0, 1, 0], rotation: [0, 0, 0] },
			{ name: 'Z', color: 0x4444ff, direction: [0, 0, 1], rotation: [Math.PI / 2, 0, 0] }
		];

		this.axisLabels = [];

		for (const axis of axes) {
			const [x, y, z] = axis.direction;

			// Create axis line
			const line = new THREE.Line(
				new THREE.BufferGeometry().setFromPoints([
					new THREE.Vector3(0, 0, 0),
					new THREE.Vector3(x * axisLength, y * axisLength, z * axisLength)
				]),
				new THREE.LineBasicMaterial({ color: axis.color })
			);
			this.axisGroup.add(line);

			// Create arrow head
			const arrow = new THREE.Mesh(
				new THREE.ConeGeometry(arrowRadius, arrowLength, 8),
				new THREE.MeshBasicMaterial({ color: axis.color })
			);
			arrow.position.set(x * axisLength, y * axisLength, z * axisLength);
			arrow.rotation.set(...axis.rotation);
			this.axisGroup.add(arrow);

			// Create label
			const midAxisPos = 0.65;
			const label = new THREE.Sprite(
				new THREE.SpriteMaterial({
					map: this.createTextTexture(`+${axis.name}`, `#${axis.color.toString(16).padStart(6, '0')}`),
					transparent: true,
					alphaTest: 0.1
				})
			);
			label.position.set(x * midAxisPos, y * midAxisPos, z * midAxisPos);
			label.scale.set(CONFIG.navball.labelScale, CONFIG.navball.labelScale, 1);
			this.axisGroup.add(label);
			this.axisLabels.push(label);
		}
	}

	createTextTexture(text, color) {
		const canvas = document.createElement('canvas');
		canvas.width = 128;
		canvas.height = 128;
		const ctx = canvas.getContext('2d');

		ctx.fillStyle = color;
		ctx.font = 'bold 64px Arial';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		ctx.fillText(text, 64, 64);

		return new THREE.CanvasTexture(canvas);
	}

	setupControls() {
		this.renderer.domElement.addEventListener('mousedown', (e) => {
			this.isDragging = true;
			this.lastMouseX = e.clientX;
			this.lastMouseY = e.clientY;
			this.renderer.domElement.style.cursor = 'grabbing';
		});

		document.addEventListener('mousemove', (e) => {
			if (!this.isDragging) return;

			const deltaX = e.clientX - this.lastMouseX;
			const deltaY = e.clientY - this.lastMouseY;

			this.yaw += deltaX * this.sensitivity;
			this.pitch += deltaY * this.sensitivity;
			this.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.pitch));

			this.lastMouseX = e.clientX;
			this.lastMouseY = e.clientY;

			this.updateRotation();
		});

		document.addEventListener('mouseup', () => {
			this.isDragging = false;
			this.renderer.domElement.style.cursor = 'grab';
		});

		this.renderer.domElement.style.cursor = 'grab';
	}

	updateRotation() {
		this.navballMesh.rotation.set(0, 0, 0);
		this.navballMesh.rotateY(this.yaw);
		this.navballMesh.rotateX(this.pitch);

		this.axisGroup.rotation.set(0, 0, 0);
		this.axisGroup.rotateY(this.yaw);
		this.axisGroup.rotateX(this.pitch);
	}

	animate() {
		requestAnimationFrame(() => this.animate());
		this.renderer.render(this.scene, this.camera);
	}

	// API methods for external control
	setRotation(yaw, pitch) {
		this.yaw = yaw;
		this.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch));
		this.updateRotation();
	}

	getRotation() {
		return { yaw: this.yaw, pitch: this.pitch };
	}

	// Apply inverted camera quaternion to show world orientation from camera perspective
	syncWithCameraQuaternion(quaternion) {
		// Invert the quaternion: q* = (w, -x, -y, -z) for unit quaternions
		const invertedQuaternion = new THREE.Quaternion(
			-quaternion.x,
			-quaternion.y,
			-quaternion.z,
			quaternion.w
		);

		// Apply inverted quaternion to show world axes as seen from camera
		this.navballMesh.quaternion.copy(invertedQuaternion);
		this.axisGroup.quaternion.copy(invertedQuaternion);
	}
}