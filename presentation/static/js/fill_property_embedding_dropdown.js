document.addEventListener('DOMContentLoaded', function() {
    const similaritySelect = document.getElementById('propertiesSelect');

    fetch('/drugembedding/fillPropertyNames')
        .then(response => response.json())
        .then(data => {
            // Loop through the data and create option elements
            data.forEach(o => {
                const exists = Array.from(similaritySelect.options).some(option => option.value === o.name);

                if (!exists) {
                    const option = document.createElement('option');
                    option.value = o.name;
                    option.text = o.name;
                    similaritySelect.appendChild(option);
                }
            });
        })
        .catch(error => console.log('Error fetching types data:', error));
});