document.addEventListener('DOMContentLoaded', function() {

    fetch(`/drug/drugList`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {

            $('#drugsTable').DataTable({
                data: data.data,
                destroy: true,
                columns: [
                    { data: 'id' },
                    { data: 'drugbank_id',
                        render: function(data, type, row, meta) {
                            return '<a href="/drug/details/' + data + '">' + data + '</a>';
                        } },
                    { data: 'drug_name', className: 'forty-percent-width' },
                    { data: 'state' }
                ],
                paging: true,
                searching: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                language: {
                    search: "_INPUT_",
                    searchPlaceholder: "Search drugs..."
                }
            });
        } else {
            console.log('Error: No data found.');
        }
    })
    .catch(error => console.log('Error:', error));
});
