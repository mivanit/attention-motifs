/**
 * PNG Loader Module
 * Loads PNG images and extracts pixel values as attention matrices
 */

async function loadPNGAsMatrix(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        
        img.onload = function() {
            // Create canvas to extract pixel data
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = img.width;
            canvas.height = img.height;
            
            // Draw image to canvas
            ctx.drawImage(img, 0, 0);
            
            // Get pixel data
            const imageData = ctx.getImageData(0, 0, img.width, img.height);
            const pixels = imageData.data;
            
            // Extract grayscale values (using R channel since image is grayscale)
            // Convert from 0-255 to 0-1 range
            const matrix = new Float32Array(img.width * img.height);
            
            for (let i = 0; i < matrix.length; i++) {
                // Each pixel has 4 values (RGBA), we just need R
                matrix[i] = pixels[i * 4] / 255.0;
            }
            
            resolve({
                data: matrix,
                width: img.width,
                height: img.height
            });
        };
        
        img.onerror = function() {
            reject(new Error(`Failed to load image: ${url}`));
        };
        
        img.src = url;
    });
}