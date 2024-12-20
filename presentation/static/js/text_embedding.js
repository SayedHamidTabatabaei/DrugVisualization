function filterEmbeddings()
{
    let text_type = '';
    let embedding_type = '';

    try {

        text_type = document.getElementById('propertiesSelect').value;
        embedding_type = document.querySelector('input[name="embeddingRadioOption"]:checked').value;

        if (text_type === '') {
            throw new Error('Similarity is required');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    showSpinner();

    function fill_interaction_embedding_table(input_data) {
        document.getElementById('embeddingGrid').style.display = 'none'
        document.getElementById('interactionEmbeddingGrid').style.display = 'block'

        $('#interactionEmbeddingsTable').DataTable({
            data: input_data,
            destroy: true,
            serverSide: true,
            ajax: {
                url: '/drugembedding/textEmbedding', // The URL to fetch data from
                type: 'GET', // HTTP method (GET or POST)
                data: function (d) {
                    d.text_type = text_type;
                    d.embedding_type = embedding_type;
                },
                error: function (xhr, error, thrown) {
                    console.error('Error fetching data from the server:', error);
                }
            },
            columns: [
                { data: 'id' },
                { data: 'drugbank1_id' },
                { data: 'drug1_name' },
                { data: 'drugbank2_id' },
                { data: 'drug2_name' },
                {
                    data: 'embedding',
                    className: 'forty-percent-width',
                    render: function (data, type, row) {
                        if (type === 'display') {
                            return data.length > 300 ? data.substr(0, 300) + '...' : data;
                        }
                        return data; // Return the full value for other data types like export or sorting
                    }
                },
                {
                    data: 'text' ,
                    className: 'forty-percent-width',
                    render: function (data, type, row) {
                        if (type === 'display') {
                            return data.length > 300 ? data.substr(0, 300) + '...' : data;
                        }
                        return data; // Return the full value for other data types like export or sorting
                    }
                }
            ],
            paging: true,
            searching: true,
            ordering: true,
            lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
            language: {
                search: "_INPUT_",
                searchPlaceholder: "Search ..."
            },
            initComplete: function() {
                let table = document.getElementById('interactionEmbeddingsTable');
                if (table) {
                    table.removeAttribute('style');
                }
            }
        });
    }

    function fill_drug_embedding_table(input_data) {
        document.getElementById('interactionEmbeddingGrid').style.display = 'none'
        document.getElementById('embeddingGrid').style.display = 'block'

        $('#embeddingsTable').DataTable({
            data: input_data,
            destroy: true,
            serverSide: true,
            ajax: {
                url: '/drugembedding/textEmbedding', // The URL to fetch data from
                type: 'GET', // HTTP method (GET or POST)
                data: function (d) {
                    d.text_type = text_type;
                    d.embedding_type = embedding_type;
                },
                error: function (xhr, error, thrown) {
                    console.error('Error fetching data from the server:', error);
                }
            },
            columns: [
                { data: 'id' },
                { data: 'drugbank_id' },
                { data: 'drug_name' },
                {
                    data: 'embedding',
                    className: 'forty-percent-width',
                    render: function (data, type, row) {
                        if (type === 'display') {
                            return data.length > 300 ? data.substr(0, 300) + '...' : data;
                        }
                        return data; // Return the full value for other data types like export or sorting
                    }
                },
                {
                    data: 'text' ,
                    className: 'forty-percent-width',
                    render: function (data, type, row) {
                        if (type === 'display') {
                            return data.length > 300 ? data.substr(0, 300) + '...' : data;
                        }
                        return data; // Return the full value for other data types like export or sorting
                    }
                }
            ],
            paging: true,
            searching: true,
            ordering: true,
            lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
            language: {
                search: "_INPUT_",
                searchPlaceholder: "Search ..."
            },
            initComplete: function() {
                let table = document.getElementById('embeddingsTable');
                if (table) {
                    table.removeAttribute('style');
                }
            }
        });
    }

    fetch(`/drugembedding/textEmbedding?start=0&length=10&text_type=${text_type}&embedding_type=${embedding_type}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            if (text_type === 'InteractionDescription') {
                fill_interaction_embedding_table(data.data);
            }
            else{
                fill_drug_embedding_table(data.data);
            }

        } else {
            console.log('Error: No data found.');
        }

        hideSpinner(true);
    })
    .catch(error => {
        console.log('Error:', error);
        hideSpinner(false);
    });
}

function calculateEmbeddings()
{
    let text_type = '';
    let embedding_type = '';

    try {

        text_type = document.getElementById('propertiesSelect').value;
        embedding_type = document.querySelector('input[name="embeddingRadioOption"]:checked').value;

        if (text_type === '') {
            throw new Error('Similarity is required');
        }
    } catch (ex){
        alert(`Please select the input fields! \n ${ex}`)
        return false;
    }

    showSpinner();

    fetch(`/drugembedding/calculateEmbedding?text_type=${text_type}&embedding_type=${embedding_type}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        hideSpinner(true);
    })
    .catch(error => {
        console.error('Error:', error);
        hideSpinner(false);
    });
}
