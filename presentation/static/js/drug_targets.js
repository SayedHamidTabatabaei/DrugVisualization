document.addEventListener('DOMContentLoaded', function() {
    fetch('/target/drugTargets?start=0&length=10')
        .then(response => response.json())
        .then(data => {

            const tableHeaders = document.getElementById('tableHeaders');
            data.columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                tableHeaders.appendChild(th);
            });

            $('#drug_target_table').DataTable({
                scrollX: true,
                scrollY: 400,
                fixedColumns: {
                    leftColumns: 1,
                },
                serverSide: true,
                ajax: function(data, callback, settings) {
                    const params = {
                        start: data.start,
                        length: data.length,
                        draw: data.draw
                    };
                    fetch(`/target/drugTargets?start=${params.start}&length=${params.length}&draw=${params.draw}`)
                        .then(response => response.json())
                        .then(data => {
                            callback(data);
                        })
                        .catch(error => console.log('Error fetching data:', error));
                },
                columns: data.columns.map(col => ({ data: col })),
                paging: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, 100000], [10, 25, 50, "All"]]
            });
        })
        .catch(error => console.log('Error fetching data:', error));
});
