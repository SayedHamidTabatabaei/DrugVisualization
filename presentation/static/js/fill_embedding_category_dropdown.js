function fillEmbeddingSelect(select_id) {
    const embeddingSelect = document.getElementById(select_id);

    fetch('/drugembedding/fillEmbeddingCategories')
        .then(response => response.json())
        .then(data => {
            // Loop through the data and create option elements
            data.forEach(o => {
                const exists = Array.from(embeddingSelect.options).some(option => option.value === o.name);

                if (!exists) {
                    const option = document.createElement('option');
                    option.value = o.name;
                    option.text = o.name;
                    embeddingSelect.appendChild(option);
                }
            });
        })
        .catch(error => console.log('Error fetching types data:', error));
}