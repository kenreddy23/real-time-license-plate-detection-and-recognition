function processImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image first');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading state
    const platesContainer = document.getElementById('platesContainer');
    platesContainer.innerHTML = '<p>Processing...</p>';

    fetch('/detect', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display results
        platesContainer.innerHTML = '';
        
        if (data.error) {
            platesContainer.innerHTML = `<p class="error">${data.error}</p>`;
            return;
        }

        // Show image preview
        const imagePreview = document.getElementById('imagePreview');
        imagePreview.innerHTML = `
            <img src="/uploads/${data.filename}" alt="Uploaded Image">
        `;

        // Show license plates
        data.plates.forEach(plate => {
            const plateDiv = document.createElement('div');
            plateDiv.className = 'plate-result';
            plateDiv.textContent = `Detected Plate: ${plate}`;
            platesContainer.appendChild(plateDiv);
        });
    })
    .catch(error => {
        platesContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    });
}