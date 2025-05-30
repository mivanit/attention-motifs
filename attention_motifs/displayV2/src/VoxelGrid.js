class VoxelGrid {
	constructor(voxelSize = 20, worldSize = 500) {
		this.voxelSize = voxelSize;
		this.worldSize = worldSize;
		this.voxelsPerSide = Math.ceil(this.worldSize * 2 / this.voxelSize);
		this.voxelGrid = new Map();
		this.activeVoxels = new Set();
	}

	clear() {
		this.voxelGrid.clear();
		this.activeVoxels.clear();
	}

	getVoxelKey(position) {
		const x = Math.floor((position.x + this.worldSize) / this.voxelSize);
		const y = Math.floor((position.y + this.worldSize) / this.voxelSize);
		const z = Math.floor((position.z + this.worldSize) / this.voxelSize);
		return `${x},${y},${z}`;
	}

	getVoxelCenter(voxelKey) {
		const [x, y, z] = voxelKey.split(',').map(Number);
		return new THREE.Vector3(
			(x + 0.5) * this.voxelSize - this.worldSize,
			(y + 0.5) * this.voxelSize - this.worldSize,
			(z + 0.5) * this.voxelSize - this.worldSize
		);
	}

	distributePoints(points) {
		this.voxelGrid.clear();

		points.forEach(point => {
			const voxelKey = this.getVoxelKey(point.position);

			if (!this.voxelGrid.has(voxelKey)) {
				this.voxelGrid.set(voxelKey, {
					points: [],
					mesh: null,
					center: this.getVoxelCenter(voxelKey)
				});
			}

			this.voxelGrid.get(voxelKey).points.push(point);
		});
	}

	getVoxel(voxelKey) {
		return this.voxelGrid.get(voxelKey);
	}

	getAllVoxels() {
		return this.voxelGrid;
	}

	clearMeshes(scene) {
		this.activeVoxels.forEach(voxelKey => {
			const voxel = this.voxelGrid.get(voxelKey);
			if (voxel && voxel.mesh) {
				scene.remove(voxel.mesh);
				if (voxel.mesh.geometry) voxel.mesh.geometry.dispose();
				if (voxel.mesh.material) voxel.mesh.material.dispose();
				voxel.mesh = null;
			}
		});
		this.activeVoxels.clear();
	}

	setActiveVoxels(voxelKeys) {
		this.activeVoxels = new Set(voxelKeys);
	}

	getActiveVoxels() {
		return this.activeVoxels;
	}
}