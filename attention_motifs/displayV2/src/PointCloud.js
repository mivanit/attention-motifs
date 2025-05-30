class PointCloud {
    constructor() {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        
        this.points = null;
        this.settings = {
            pointSize: 3.0,
            opacity: 0.8,
            speed: 10,
            pointCount: 25000
        };
        
        // Movement
        this.keys = {};
        this.mouseX = 0;
        this.mouseY = 0;
        this.roll = 0;
        this.velocity = new THREE.Vector3();
        
        this.init();
    }
    
    init() {
        this.setupRenderer();
        this.setupControls();
        this.setupMovement();
        this.generatePoints();
        this.animate();
    }
    
    setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000011);
        document.getElementById('container').appendChild(this.renderer.domElement);
        this.camera.position.set(0, 0, 0);
    }
    
    setupControls() {
        const controls = {
            pointSize: { element: document.getElementById('pointSize'), display: document.getElementById('pointSizeValue') },
            opacity: { element: document.getElementById('opacity'), display: document.getElementById('opacityValue') },
            speed: { element: document.getElementById('speed'), display: document.getElementById('speedValue') },
            pointCount: { element: document.getElementById('pointCount'), display: document.getElementById('pointCountValue') }
        };
        
        Object.keys(controls).forEach(key => {
            const control = controls[key];
            control.element.addEventListener('input', () => {
                const value = parseFloat(control.element.value);
                this.settings[key] = value;
                control.display.textContent = value;
                this.handleSettingChange(key, value);
            });
        });
        
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }
    
    setupMovement() {
        document.addEventListener('keydown', (e) => {
            this.keys[e.code] = true;
        });
        
        document.addEventListener('keyup', (e) => {
            this.keys[e.code] = false;
        });
        
        document.addEventListener('mousemove', (e) => {
            if (document.pointerLockElement === document.body) {
                this.mouseX -= e.movementX * 0.002;
                this.mouseY -= e.movementY * 0.002;
                this.mouseY = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.mouseY));
            }
        });
        
        document.addEventListener('dblclick', () => {
            if (document.pointerLockElement === document.body) {
                document.exitPointerLock();
            } else {
                document.body.requestPointerLock();
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Escape' && document.pointerLockElement === document.body) {
                document.exitPointerLock();
            }
        });
    }
    
    generatePoints() {
        if (this.points) {
            this.scene.remove(this.points);
            this.points.geometry.dispose();
            this.points.material.dispose();
        }
        
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        
        const range = 1000;
        
        for (let i = 0; i < this.settings.pointCount; i++) {
            // Random positions in 3D space
            positions.push(
                (Math.random() - 0.5) * range,
                (Math.random() - 0.5) * range,
                (Math.random() - 0.5) * range
            );
            
            // Random colors
            const color = new THREE.Color().setHSL(Math.random(), 0.7, 0.6);
            colors.push(color.r, color.g, color.b);
        }
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: this.settings.pointSize,
            opacity: this.settings.opacity,
            transparent: this.settings.opacity < 1,
            vertexColors: true,
            sizeAttenuation: true
        });
        
        this.points = new THREE.Points(geometry, material);
        this.scene.add(this.points);
        
        document.getElementById('renderedCount').textContent = this.settings.pointCount;
    }
    
    handleSettingChange(property, value) {
        if (property === 'pointCount') {
            this.generatePoints();
        } else if (property === 'pointSize' || property === 'opacity') {
            if (this.points && this.points.material) {
                this.points.material.size = this.settings.pointSize;
                this.points.material.opacity = this.settings.opacity;
                this.points.material.transparent = this.settings.opacity < 1;
                this.points.material.needsUpdate = true;
            }
        }
    }
    
    updateMovement() {
        // Roll rotation with Q/E
        if (this.keys['KeyQ']) this.roll += 0.02;
        if (this.keys['KeyE']) this.roll -= 0.02;
        
        // Apply rotation in order: Y (yaw), X (pitch), Z (roll)
        this.camera.rotation.set(this.mouseY, this.mouseX, this.roll, 'YXZ');
        
        // Movement relative to camera orientation
        this.velocity.set(0, 0, 0);
        
        if (this.keys['KeyW']) this.velocity.z -= 1;  // Forward (negative Z in camera space)
        if (this.keys['KeyS']) this.velocity.z += 1;  // Back
        if (this.keys['KeyA']) this.velocity.x -= 1;  // Left
        if (this.keys['KeyD']) this.velocity.x += 1;  // Right
        
        const speedMultiplier = this.keys['ShiftLeft'] ? 3 : 1;
        this.velocity.normalize().multiplyScalar(this.settings.speed * speedMultiplier * 0.016);
        
        // Transform movement vector by camera rotation
        this.velocity.applyQuaternion(this.camera.quaternion);
        
        // Apply movement to camera position
        this.camera.position.add(this.velocity);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.updateMovement();
        this.renderer.render(this.scene, this.camera);
    }
}