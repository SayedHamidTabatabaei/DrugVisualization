document.addEventListener('DOMContentLoaded', function() {
    const similaritySelect = document.getElementById('similaritySelect');

    if (!similaritySelect) {return false;}

    fetch('/similarity/fillSimilarityTypes')
        .then(response => response.json())
        .then(data => {
            // Loop through the data and create option elements
            data.forEach(o => {
                const option = document.createElement('option');
                option.value = o.name;
                option.text = o.name;
                similaritySelect.appendChild(option);
            });
        })
        .catch(error => console.log('Error fetching types data:', error));
});

function fillSimilaritySelect(select_id) {
    const similaritySelect = document.getElementById(select_id);

    fetch('/similarity/fillSimilarityTypes')
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
}

function filterSimilaritySelect(select_id, checkbox_id) {

    const similaritySelect = document.getElementById(select_id);
    const checkbox = document.getElementById(checkbox_id);

    if (!checkbox || !checkbox.checked) {return false;}

    const category = checkbox.value;

    fetch(`/similarity/filterUsageSimilarityTypes?category=${category}`)
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
}
