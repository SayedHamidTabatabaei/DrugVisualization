document.addEventListener('DOMContentLoaded', function() {
    const similaritySelect = document.getElementById('trainModelSelect');

    fetch('/training/fillTrainingModels')
        .then(response => response.json())
        .then(data => {
            // Loop through the data and create option elements
            data.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.text = model.name;
                similaritySelect.appendChild(option);
            });
        })
        .catch(error => console.log('Error fetching types data:', error));
});