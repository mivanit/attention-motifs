class PointCloudExplorer {
	constructor() {
		this.scene = new THREE.Scene();
		this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
		this.renderer = new THREE.WebGLRenderer({ antialias: true });
		
		this.voxelGrid = new VoxelGrid(20, 500);
		this.controls = new Controls(this.camera);
		this.points = [];
		this.frameCount = 0;

		this.init();
	}

	init() {
		this.setupRenderer();
		this.setupCallbacks();
		this.generatePoints();
		this.animate();
	}

	setupRenderer() {
		this.renderer.setSize(window.innerWidth, window.innerHeight);
		this.renderer.setClearColor(0x000011);
		document.getElementById('container').appendChild(this.renderer.domElement);
		this.camera.position.set(0, 0, 0);
	}

	setupCallbacks() {
		this.controls.setOnSettingChange((property, value) => {
			this.handleSettingChange(property, value);
		});

		this.controls.setOnWindowResize(() => {
			this.renderer.setSize(window.innerWidth, window.innerHeight);
		});

		this.controls.setOnMouseMove((event) => {
			this.controls.checkHover(event, this.voxelGrid.getActiveVoxels(), this.voxelGrid.getAllVoxels());
		});
	}

	generatePoints() {
		this.points = [];
		const count = this.controls.settings.pointCount;

		for (let i = 0; i < count; i++) {
			const point = {
				id: i,
				position: new THREE.Vector3(
					(Math.random() - 0.5) * this.voxelGrid.worldSize * 2,
					(Math.random() - 0.5) * this.voxelGrid.worldSize * 2,
					(Math.random() - 0.5) * this.voxelGrid.worldSize * 2
				),
				color: new THREE.Color().setHSL(Math.random(), 0.7, 0.6)
			};
			this.points.push(point);
		}

		this.voxelGrid.distributePoints(this.points);
		this.updateLOD();
	}

	updateLOD() {
		const cameraPos = this.camera.position;
		const newActiveVoxels = [];
		let totalRenderedPoints = 0;

		// Clear old meshes
		this.voxelGrid.clearMeshes(this.scene);

		// Create new meshes for visible voxels
		this.voxelGrid.getAllVoxels().forEach((voxel, voxelKey) => {
			const distance = cameraPos.distanceTo(voxel.center);

			if (distance < this.controls.settings.lodDistance * 3) {
				newActiveVoxels.push(voxelKey);

				const geometry = new THREE.BufferGeometry();
				const positions = [];
				const colors = [];

				// LOD logic: use all points if close, subsample if far
				let pointsToRender = voxel.points;
				if (distance > this.controls.settings.lodDistance) {
					const lodFactor = Math.max(0.05, this.controls.settings.lodDistance / distance);
					const step = Math.ceil(1 / lodFactor);
					pointsToRender = voxel.points.filter((_, i) => i % step === 0);
				}

				pointsToRender.forEach(point => {
					positions.push(point.position.x, point.position.y, point.position.z);
					colors.push(point.color.r, point.color.g, point.color.b);
				});

				if (positions.length > 0) {
					totalRenderedPoints += pointsToRender.length;

					geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
					geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

					const material = new THREE.PointsMaterial({
						size: this.controls.settings.pointSize,
						opacity: this.controls.settings.opacity,
						transparent: this.controls.settings.opacity < 1,
						vertexColors: true,
						sizeAttenuation: true
					});

					voxel.mesh = new THREE.Points(geometry, material);
					voxel.mesh.userData = { voxelKey, points: pointsToRender };
					this.scene.add(voxel.mesh);
				}
			}
		});

		this.voxelGrid.setActiveVoxels(newActiveVoxels);
		document.getElementById('renderedCount').textContent = totalRenderedPoints;
	}

	handleSettingChange(property, value) {
		if (property === 'pointCount') {
			// Regenerate all points
			this.voxelGrid.clearMeshes(this.scene);
			this.voxelGrid.clear();
			this.generatePoints();
		} else if (property === 'pointSize' || property === 'opacity') {
			// Update existing materials
			this.voxelGrid.getActiveVoxels().forEach(voxelKey => {
				const voxel = this.voxelGrid.getVoxel(voxelKey);
				if (voxel && voxel.mesh && voxel.mesh.material) {
					voxel.mesh.material.size = this.controls.settings.pointSize;
					voxel.mesh.material.opacity = this.controls.settings.opacity;
					voxel.mesh.material.transparent = this.controls.settings.opacity < 1;
					voxel.mesh.material.needsUpdate = true;
				}
			});
		} else {
			// Refresh LOD for other settings
			this.updateLOD();
		}
	}

	animate() {
		requestAnimationFrame(() => this.animate());

		this.controls.updateMovement();

		// Only update LOD every few frames for performance, or when movement occurs
		if (this.frameCount % 3 === 0 || this.controls.isMoving()) {
			this.updateLOD();
		}
		this.frameCount++;

		this.renderer.render(this.scene, this.camera);
	}
}