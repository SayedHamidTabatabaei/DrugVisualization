document.addEventListener('DOMContentLoaded', function() {

    const train_history_id = document.getElementById('train-history-id').textContent;

    showSpinner();

    fetch(`/training/get_history_details?trainHistoryId=${train_history_id}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {

            $('#trainHistoryDetailsTable').DataTable({
                data: data.data,
                destroy: true,
                columns: [
                    { data: 'training_label' },
                    { data: 'f1_score' },
                    { data: 'accuracy' },
                    { data: 'auc' },
                    { data: 'aupr' }
                ],
                paging: false,
                searching: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                language: {
                    search: "_INPUT_",
                    searchPlaceholder: "Search ..."
                }
            });
        } else {
            console.log('Error: No data found.');
        }
        hideSpinner(true);
    })
    .catch(error => {
        console.log('Error:', error)
        hideSpinner(false);
    });
});