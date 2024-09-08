document.addEventListener('DOMContentLoaded', function() {

    fetch('/drugembedding/fillEmbeddingCategories') // URL to your server API endpoint
        .then(response => response.json())
        .then(data => {
            const radioButtonsContainer = document.getElementById('embeddingCategoryRadioContainer');

            // Clear any existing radio buttons
            radioButtonsContainer.innerHTML = '';

            // Iterate over the data and create radio buttons
            data.forEach(option => {
                const radioWrapper = document.createElement('div');
                radioWrapper.classList.add('radio-option');

                const radioButton = document.createElement('input');
                radioButton.type = 'radio';
                radioButton.id = 'embedding' + option.id;
                radioButton.name = 'embeddingRadioOption';
                radioButton.value = option.value;

                const label = document.createElement('label');
                label.htmlFor = 'embedding' + option.id;
                label.textContent = option.label;

                // Append radio button and label to the wrapper
                radioWrapper.appendChild(radioButton);
                radioWrapper.appendChild(label);

                // Append wrapper to the container
                radioButtonsContainer.appendChild(radioWrapper);
            });
        })
        .catch(error => {
            console.error('Error fetching radio button data:', error);
        })
        .finally(() => {
        });
});