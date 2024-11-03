document.addEventListener('DOMContentLoaded', function() {

    var submitBtn = document.getElementById('submit-btn');

    if (submitBtn) {
        submitBtn.addEventListener('click', function() {
            var drugbank_id = document.getElementById('drugbank-id').value;

            function visualizeMolecule(rdkitMol) {
                    var viewer = $3Dmol.createViewer('mol-container', {
                        defaultcolors: $3Dmol.rasmolElementColors
                    });

                    viewer.addModel(rdkitMol, 'sdf');

                    viewer.setStyle({}, {stick:{}});

                    viewer.zoomTo();

                    viewer.render();
                }

            fetch(`/drug/visualization/${encodeURIComponent(drugbank_id)}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {

                var resultContainer = document.getElementById('result');

                if (data.status) {

                    resultContainer.innerHTML = '';
                    visualizeMolecule(data.mol_block);
                } else {
                    var mol_container = document.getElementById('mol-container');
                    mol_container.innerHTML = '';
                    resultContainer.innerHTML = `<p>${data.message}</p>`;
                }
            })
            .catch(error => console.log('Error:'));
        });
    }
});
