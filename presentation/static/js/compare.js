document.addEventListener('DOMContentLoaded', function() {

    get_history();
});

let imageGroups = {}

function get_history()
{
    const scenario = document.getElementById('scenarioSelect').value;
    const train_model = document.getElementById('trainModelSelect').value;
    const dateFilter = document.getElementById('dateFilter').value;
    const min_sample_count = document.getElementById('minSampleCountSelect').value;

    showSpinner();

    fetch(`/training/get_history?start=0&length=10000&scenario=${scenario}&trainModel=${train_model}&date=${dateFilter}&min_sample_count=${min_sample_count}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {

            data.data.forEach(item => {item.is_checked = false})

            const maxAccuracy = Math.max(...data.data.map(item => item.accuracy));
            const minLoss = Math.min(...data.data.map(item => item.loss));
            const maxF1Score_Weighted = Math.max(...data.data.map(item => item.f1_score_weighted));
            const maxF1Score_Micro = Math.max(...data.data.map(item => item.f1_score_micro));
            const maxF1Score_Macro = Math.max(...data.data.map(item => item.f1_score_macro));
            const maxAuc_Weighted = Math.max(...data.data.map(item => item.auc_weighted));
            const maxAuc_Micro = Math.max(...data.data.map(item => item.auc_micro));
            const maxAuc_Macro = Math.max(...data.data.map(item => item.auc_macro));
            const maxAupr_Weighted = Math.max(...data.data.map(item => item.aupr_weighted));
            const maxAupr_Micro = Math.max(...data.data.map(item => item.aupr_micro));
            const maxAupr_Macro = Math.max(...data.data.map(item => item.aupr_macro));
            const maxRecall_Weighted = Math.max(...data.data.map(item => item.recall_weighted));
            const maxRecall_Micro = Math.max(...data.data.map(item => item.recall_micro));
            const maxRecall_Macro = Math.max(...data.data.map(item => item.recall_macro));
            const maxPrecision_Weighted = Math.max(...data.data.map(item => item.precision_weighted));
            const maxPrecision_Micro = Math.max(...data.data.map(item => item.precision_micro));
            const maxPrecision_Macro = Math.max(...data.data.map(item => item.precision_macro));

            $('#selectableTrainTable').DataTable({
                data: data.data,
                destroy: true,
                paging: false,
                columns: [
                    {
                        data: null,
                        render: function(data, type, row, meta) {
                            const checkboxValue = `${row.id}|${row.name}`;
                            const isChecked = row.is_checked ? 'checked' : '';

                            return '<label class="checkbox-label" style="width: 30px;">\n' +
                                `    <input type="checkbox" id="row-check" class="row-checkbox" ${isChecked} onclick="compare_trainings();updateRowState(${meta.row}, this.checked);" value="${checkboxValue}">\n` +
                                '    <span class="checkbox-custom" style="margin-top: 0;"></span>' +
                                '</label>'
                        },
                        orderable: false,
                        searchable: false
                    },
                    {
                        className: 'check-text-column',
                        data: 'is_checked'
                    },
                    {
                        className: 'id-column',
                        data: 'id'
                    },
                    {
                        className: 'total-name-column',
                        render: function(data, type, row) {
                            return `<span title="${row.description}">${row.name}</span><span class="inside-train-model"><br/>(${row.train_model})</span><span class="inside-description"><br/>(${row.description})</span>`;
                        }
                    },
                    {
                        className: 'name-column',
                        data: 'name'
                    },
                    {
                        className: 'train-model-column',
                        data: 'train_model'
                    },
                    {
                        className: 'description-column',
                        data: 'description'
                    },
                    {
                        className: 'loss-column',
                        render: function(data, type, row) {
                            let class_weight = ""
                            if(row.class_weight) {
                                class_weight = "<br/><span>(Class Weight)</span>"
                            }

                            let loss_function = "Default"
                            if(row.loss_function) {
                                loss_function = row.loss_function
                            }

                            return `<span>${loss_function}</span>${class_weight}`;
                        }
                    },
                    {
                        data: 'accuracy',
                        render: function(data, type, row) {
                            const formatted_accuracy = row.accuracy ? row.accuracy.toFixed(2) : '0.00';

                            if (row.accuracy === maxAccuracy) {

                                return '<strong>' + formatted_accuracy + '</strong>';
                            } else {
                                return formatted_accuracy;
                            }
                        }
                    },
                    {
                        data: 'loss',
                        render: function(data, type, row) {
                            const formatted_loss = row.loss ? row.loss.toFixed(2) : '0.00';

                            if (row.loss === minLoss) {
                                return '<strong>' + formatted_loss + '</strong>';
                            } else {
                                return formatted_loss;
                            }
                        }
                    },
                    {
                        data: 'f1_score_weighted',
                        render: function(data, type, row) {
                            const formatted_f1_score_weighted = row.f1_score_weighted ? row.f1_score_weighted.toFixed(2) : '0.00';

                            if (row.f1_score_weighted === maxF1Score_Weighted) {
                                return '<strong>' + formatted_f1_score_weighted + '</strong>';
                            } else {
                                return formatted_f1_score_weighted;
                            }
                        }
                    },
                    {
                        data: 'f1_score_micro',
                        render: function(data, type, row) {
                            const formatted_f1_score_micro = row.f1_score_micro ? row.f1_score_micro.toFixed(2) : '0.00';

                            if (row.f1_score_micro === maxF1Score_Micro) {
                                return '<strong>' + formatted_f1_score_micro + '</strong>';
                            } else {
                                return formatted_f1_score_micro;
                            }
                        }
                    },
                    {
                        data: 'f1_score_macro',
                        render: function(data, type, row) {
                            const formatted_f1_score_macro = row.f1_score_macro ? row.f1_score_macro.toFixed(2) : '0.00';

                            if (row.f1_score_macro === maxF1Score_Macro) {
                                return '<strong>' + formatted_f1_score_macro + '</strong>';
                            } else {
                                return formatted_f1_score_macro;
                            }
                        }
                    },
                    {
                        data: 'auc_weighted',
                        render: function(data, type, row) {
                            const formatted_auc_weighted = row.auc_weighted ? row.auc_weighted.toFixed(2) : '0.00';

                            if (row.auc_weighted === maxAuc_Weighted) {
                                return '<strong>' + formatted_auc_weighted + '</strong>';
                            } else {
                                return formatted_auc_weighted;
                            }
                        }
                    },
                    {
                        data: 'auc_micro',
                        render: function(data, type, row) {
                            const formatted_auc_micro = row.auc_micro ? row.auc_micro.toFixed(2) : '0.00';

                            if (row.auc_micro === maxAuc_Micro) {
                                return '<strong>' + formatted_auc_micro + '</strong>';
                            } else {
                                return formatted_auc_micro;
                            }
                        }
                    },
                    {
                        data: 'auc_macro',
                        render: function(data, type, row) {
                            const formatted_auc_macro = row.auc_macro ? row.auc_macro.toFixed(2) : '0.00';

                            if (row.auc_macro === maxAuc_Macro) {
                                return '<strong>' + formatted_auc_macro + '</strong>';
                            } else {
                                return formatted_auc_macro;
                            }
                        }
                    },
                    {
                        data: 'aupr_weighted',
                        render: function(data, type, row) {
                            const formatted_aupr_weighted = row.aupr_weighted ? row.aupr_weighted.toFixed(2) : '0.00';

                            if (row.aupr_weighted === maxAupr_Weighted) {
                                return '<strong>' + formatted_aupr_weighted + '</strong>';
                            } else {
                                return formatted_aupr_weighted;
                            }
                        }
                    },
                    {
                        data: 'aupr_micro',
                        render: function(data, type, row) {
                            const formatted_aupr_micro = row.aupr_micro ? row.aupr_micro.toFixed(2) : '0.00';

                            if (row.aupr_micro === maxAupr_Micro) {
                                return '<strong>' + formatted_aupr_micro + '</strong>';
                            } else {
                                return formatted_aupr_micro;
                            }
                        }
                    },
                    {
                        data: 'aupr_macro',
                        render: function(data, type, row) {
                            const formatted_aupr_macro = row.aupr_macro ? row.aupr_macro.toFixed(2) : '0.00';

                            if (row.aupr_macro === maxAupr_Macro) {
                                return '<strong>' + formatted_aupr_macro + '</strong>';
                            } else {
                                return formatted_aupr_macro;
                            }
                        }
                    },
                    {
                        data: 'recall_weighted',
                        render: function(data, type, row) {
                            const formatted_recall_weighted = row.recall_weighted ? row.recall_weighted.toFixed(2) : '0.00';

                            if (row.recall_weighted === maxRecall_Weighted) {
                                return '<strong>' + formatted_recall_weighted + '</strong>';
                            } else {
                                return formatted_recall_weighted;
                            }
                        }
                    },
                    {
                        data: 'recall_micro',
                        render: function(data, type, row) {
                            const formatted_recall_micro = row.recall_micro ? row.recall_micro.toFixed(2) : '0.00';

                            if (row.recall_micro === maxRecall_Micro) {
                                return '<strong>' + formatted_recall_micro + '</strong>';
                            } else {
                                return formatted_recall_micro;
                            }
                        }
                    },
                    {
                        data: 'recall_macro',
                        render: function(data, type, row) {
                            const formatted_recall_macro = row.recall_macro ? row.recall_macro.toFixed(2) : '0.00';

                            if (row.recall_macro === maxRecall_Macro) {
                                return '<strong>' + formatted_recall_macro + '</strong>';
                            } else {
                                return formatted_recall_macro;
                            }
                        }
                    },
                    {
                        data: 'precision_weighted',
                        render: function(data, type, row) {
                            const formatted_precision_weighted = row.precision_weighted ? row.precision_weighted.toFixed(2) : '0.00';

                            if (row.precision_weighted === maxPrecision_Weighted) {
                                return '<strong>' + formatted_precision_weighted + '</strong>';
                            } else {
                                return formatted_precision_weighted;
                            }
                        }
                    },
                    {
                        data: 'precision_micro',
                        render: function(data, type, row) {
                            const formatted_precision_micro = row.precision_micro ? row.precision_micro.toFixed(2) : '0.00';

                            if (row.precision_micro === maxPrecision_Micro) {
                                return '<strong>' + formatted_precision_micro + '</strong>';
                            } else {
                                return formatted_precision_micro;
                            }
                        }
                    },
                    {
                        data: 'precision_macro',
                        render: function(data, type, row) {
                            const formatted_precision_macro = row.precision_macro ? row.precision_macro.toFixed(2) : '0.00';

                            if (row.precision_macro === maxPrecision_Macro) {
                                return '<strong>' + formatted_precision_macro + '</strong>';
                            } else {
                                return formatted_precision_macro;
                            }
                        }
                    },
                    {
                        data: null,
                        orderable: false,
                        render: function (data, type, row) {
                            return `
                                <div style="min-width: 100px;">
                                    <button style="margin:1px;" class="btn btn-info" onclick="showconditions(${row.id})" data-bs-toggle="tooltip" title="Show Conditions"><i class="bi bi-file-earmark-code"></i></button>
                                    <button style="margin:1px;" class="btn btn-success" onclick="showdatareport(${row.id})" data-bs-toggle="tooltip" title="Show Data Report"><i class="bi bi-file-earmark-check"></i></button>
                                    <button style="margin:1px;" class="btn btn-success" onclick="showmodelinfo(${row.id})" data-bs-toggle="tooltip" title="Show Model Information"><i class="bi bi-file-earmark-text"></i></button>
                                    <button style="margin:1px;" class="btn btn-success" onclick="showImage('${row.train_model}')" data-bs-toggle="tooltip" title="Show Model Image"><i class="bi bi-image"></i></button>
                                </div>
                            `;
                        }
                    }
                ],
                searching: true,
                ordering: true,
                language: {
                    search: "_INPUT_",
                    searchPlaceholder: "Search ..."
                },
                initComplete: function() {
                    let table = document.getElementById('selectableTrainTable');
                    if (table) {
                        table.removeAttribute('style');
                    }

                    const nameColumn = document.querySelectorAll('.name-column');
                    nameColumn.forEach((column) => { column.style.display = 'none'; });

                    const trainingModelColumn = document.querySelectorAll('.train-model-column');
                    trainingModelColumn.forEach((column) => { column.style.display = 'none'; });

                    const descriptionColumn = document.querySelectorAll('.description-column');
                    descriptionColumn.forEach((column) => { column.style.display = 'none'; });

                    const idColumn = document.querySelectorAll('.id-column');
                    idColumn.forEach((column) => { column.style.display = 'none'; });

                },
        dom: 'Bfrtip',
        buttons: [
            {
                extend: 'pdfHtml5',
                title: 'Training Results',
                orientation: 'landscape',
                pageSize: 'A3',
                text: '<i title="PDF" class="fas fa-file-pdf"></i>',
                exportOptions: {
                    columns: ':visible'
                }
            },
            {
                extend: 'csvHtml5', // Add the CSV button
                title: 'Training Results', // Title of the CSV file
                text: '<i title="CSV" class="fas fa-file-csv"></i>', // Icon for the CSV button
                exportOptions: {
                    columns: ':visible' // Export only visible columns
                }
            },
            {
                text: '<i title="Latex Code" class="fas fa-file-code"></i>',
                action: function (e, dt, button, config) {
                    generateLatexCode(data.columns, data.data);
                }
            },
            {
                text: '<i title="Export Selected" class="fas fa-file-export"></i>',
                action: function (e, dt, button, config) {
                    exportSelected(data.columns, data.data);
                }
            },
            {
                text: '<i title="Import" class="fas fa-file-import"></i>',
                action: function (e, dt, button, config) {
                    importSelected(data.columns, data.data);
                }
            },
            {
                text: '<i title="Filter Checked" class="fas fa-filter"></i>',
                action: function (e, dt, button, config) {

                    if (button.checked === undefined){
                        button.checked = true;
                    }
                    else{
                        button.checked = !button.checked;
                    }

                    filter_checked(button.checked)

                    if (button.checked){
                        button.addClass('active-filter')
                    }
                    else{
                        button.removeClass('active-filter');
                    }
                }
            },
            {
                text: '<i title="Setting" class="fas fa-gear"></i>',
                action: function (e, dt, button, config) {

                    document.getElementById('settingModal').style.display = 'block';
                }
            }
        ]
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
}

function closeSettingModal() {
    document.getElementById('settingModal').style.display = 'none';
}

function settingSubmit(){
    const trainingModelValue = document.querySelector('input[name="trainingModel"]:checked')?.value || 'None';
    const descriptionValue = document.querySelector('input[name="description"]:checked')?.value || 'None';
    const lossValue = document.querySelector('input[name="loss"]:checked')?.value || 'None';
    const idValue = document.querySelector('input[name="id"]:checked')?.value || 'None';

    if (trainingModelValue === 'inColumn'){
        const trainingModelColumn = document.querySelectorAll('.train-model-column');
        trainingModelColumn.forEach((column) => { column.style.display = 'table-cell'; });

        const insideTrainingModel = document.querySelectorAll('.inside-train-model');
        insideTrainingModel.forEach((column) => { column.style.display = 'none'; });

    }else if (trainingModelValue === 'underName'){
        const trainingModelColumn = document.querySelectorAll('.train-model-column');
        trainingModelColumn.forEach((column) => { column.style.display = 'none'; });

        const insideTrainingModel = document.querySelectorAll('.inside-train-model');
        insideTrainingModel.forEach((column) => { column.style.display = 'inline'; });

    }else{

        const trainingModelColumn = document.querySelectorAll('.train-model-column');
        trainingModelColumn.forEach((column) => { column.style.display = 'none'; });

        const insideTrainingModel = document.querySelectorAll('.inside-train-model');
        insideTrainingModel.forEach((column) => { column.style.display = 'none'; });
    }

    if (descriptionValue === 'inColumn'){
        const descriptionColumn = document.querySelectorAll('.description-column');
        descriptionColumn.forEach((column) => { column.style.display = 'table-cell'; });

        const insideDescription = document.querySelectorAll('.inside-description');
        insideDescription.forEach((column) => { column.style.display = 'none'; });

    }else if (descriptionValue === 'underName'){
        const descriptionColumn = document.querySelectorAll('.description-column');
        descriptionColumn.forEach((column) => { column.style.display = 'none'; });

        const insideDescription = document.querySelectorAll('.inside-description');
        insideDescription.forEach((column) => { column.style.display = 'inline'; });

    }else{

        const descriptionColumn = document.querySelectorAll('.description-column');
        descriptionColumn.forEach((column) => { column.style.display = 'none'; });

        const insideDescription = document.querySelectorAll('.inside-description');
        insideDescription.forEach((column) => { column.style.display = 'none'; });
    }

    if (lossValue === 'show'){
        const lossColumn = document.querySelectorAll('.loss-column');
        lossColumn.forEach((column) => { column.style.display = 'table-cell'; });
    } else{
        const lossColumn = document.querySelectorAll('.loss-column');
        lossColumn.forEach((column) => { column.style.display = 'none'; });
    }

    if (idValue === 'show'){
        const idColumn = document.querySelectorAll('.id-column');
        idColumn.forEach((column) => { column.style.display = 'table-cell'; });
    } else{
        const idColumn = document.querySelectorAll('.id-column');
        idColumn.forEach((column) => { column.style.display = 'none'; });
    }

    if (trainingModelValue !== 'underName' && descriptionValue !== 'underName'){
        const nameColumn = document.querySelectorAll('.name-column');
        nameColumn.forEach((column) => { column.style.display = 'table-cell'; });

        const totalNameColumn = document.querySelectorAll('.total-name-column');
        totalNameColumn.forEach((column) => { column.style.display = 'none'; });

    } else {
        const nameColumn = document.querySelectorAll('.total-name-column');
        nameColumn.forEach((column) => { column.style.display = 'table-cell'; });

        const totalNameColumn = document.querySelectorAll('.name-column');
        totalNameColumn.forEach((column) => { column.style.display = 'none'; });

    }

    document.getElementById('settingModal').style.display = 'none';
}

function compare_trainings()
{
    const activeTab = document.querySelector('.tab.active');
    activeTab.click();
}

function select_all()
{
    let isChecked = document.getElementById('select-all').checked;
    let checkboxes = document.querySelectorAll('.row-checkbox');
    checkboxes.forEach(function(checkbox) {
        checkbox.checked = isChecked;
    });
}

function filter_checked(isChecked){

    let rows = document.querySelectorAll('#selectableTrainTable tbody tr'); // Select all rows once

    rows.forEach(function(row) {
        let checkbox = row.querySelector('.row-checkbox');

        if (isChecked) {
            if (checkbox && checkbox.checked) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        } else {
            row.style.display = '';
        }
    });
}

function openTab(evt, tabName) {
    // Get all elements with class="tab-content" and hide them
    const tabContents = document.getElementsByClassName("tab-content");
    const tabs = document.getElementsByClassName("tab");
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].style.display = "none";
        tabContents[i].classList.remove("active");
        tabs[i].classList.remove("active");
    }

    // Show the current tab, and add an "active" class to the tab and content
    document.getElementById(tabName).style.display = "block";
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");
}

function fill_comparing_plot(comparing_plot_type, radio_group){

    const plot_name = comparing_plot_type;

    if(radio_group){
        comparing_plot_type = document.querySelector(`input[name=${radio_group}]:checked`).value;
    }

    let selectedIds = [];
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedIds.push(id);
    });

    fetch(`/training/get_comparing_plot?trainHistoryIds=${selectedIds}&ComparePlotType=${comparing_plot_type}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            const plot = document.getElementById(`${plot_name}-plot`);

            plot.src = `data:image/png;base64,${data.image.path}`;
            plot.alt = data.image.name;
            plot.style.maxWidth = '110%';

        } else {
            console.log('Error: No data found.');
        }
        hideSpinner(true);
    })
    .catch(error => {
        console.log('Error:', error)
        hideSpinner(false);
    });
}

function fill_accuracy_plots(){
    fill_saved_plots('accuracy', 'accuracyPlotsGridContainer')
}

function updateRowState(rowIndex, isChecked){
    // Access DataTable instance
    const table = $('#selectableTrainTable').DataTable();

    // Get the row data
    const rowData = table.row(rowIndex).data();

    rowData.is_checked = isChecked;

    // Update the row in DataTables
    table.row(rowIndex).data(rowData).draw(false);
}

function create_plot_image(container, src, value) {
    const plotImage = document.createElement('img');
    plotImage.src = src;
    plotImage.alt = `Line Plot ${value}`;
    plotImage.onclick = () => openImageModal(plotImage);

    const caption = document.createElement('p');
    caption.classList.add('plot-caption');
    caption.textContent = value;

    const plotBox = document.createElement('div');
    plotBox.classList.add('plot-box');
    plotBox.appendChild(plotImage);
    plotBox.appendChild(caption);

    // Append plot box to the grid container
    container.appendChild(plotBox);
}

function fill_loss_plots() {

    fill_saved_plots('loss', 'lossPlotsGridContainer')

}
function fill_saved_plots(plotType, containerName){

    let selectedItems = new Map();
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedItems.set(id, name);
    });

    const plotsGridContainer = document.getElementById(containerName);
    plotsGridContainer.innerHTML = '';

    selectedItems.forEach((value, key) => {

        fetch(`/training/get_saved_plots?trainId=${key}&plotType=${plotType}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status) {
                const plotBox = document.createElement('div');
                const plotImage = document.createElement('img');

                if (data.data.length > 1) {
                    plotBox.classList.add('image-group');

                    plotImage.src = `/training/training_history_plots/training_plots/${key}/${data.data[0]}`;
                    plotImage.alt = `Gallery - Line Plot ${value}`;
                    plotImage.classList.add('thumbnail')
                    plotImage.onclick = () => openSlideshow(key);

                    imageGroups[key] = data.data;
                } else{
                    plotBox.classList.add('plot-box');

                    plotImage.src = `/training/training_history_plots/training_plots/${key}/loss_plot.png`;
                    plotImage.alt = `Line Plot ${value}`;
                    plotImage.onclick = () => openImageModal(plotImage);

                }

                const caption = document.createElement('p');
                caption.classList.add('plot-caption');
                caption.textContent =value;

                plotBox.appendChild(plotImage);
                plotBox.appendChild(caption);

                plotsGridContainer.appendChild(plotBox);
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
}

function fill_grid(grid_type){

    let selectedIds = [];
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedIds.push(id);
    });

    setTimeout(() => {
        fetch(`/training/get_per_category_result_details?trainHistoryIds=${selectedIds}&ComparePlotType=${grid_type}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            let grid = document.getElementById(`${grid_type}_grid`);

            let headersHtml = data.columns.map(col => `<th>${col}</th>`).join('\n');

            grid.innerHTML = `<table id='${grid_type}-table' class="display">
                                <thead>
                                    <tr id='${grid_type}-tableHeaders'>
                                        ${headersHtml}
                                    </tr>
                                </thead>
                                <tbody id='${grid_type}-tableBody'></tbody>
                            </table>`;


            // Initialize the DataTable with new data
            $(`#${grid_type}-table`).DataTable({
                destroy: true,
                data: data.data,
                scrollX: true,
                scrollY: 400,
                fixedColumns: {
                    leftColumns: 1,
                },
                columns: data.columns.map(col => ({ data: col })), // Map column names to data keys
                paging: false,
                ordering: true,
                searching: false,
                createdRow: function (row, data, dataIndex) {
                    let rowData = data; // Original row data as an object
                    let keys = Object.keys(rowData).filter(key => key !== 'level'); // Exclude 'level' column
                    let rowValues = keys.map(key => rowData[key]); // Get values for columns other than 'level'
                    let maxVal = Math.max(...rowValues);

                    // Iterate through each cell to find the max value and make it bold
                    $('td', row).each(function () {
                        if (parseFloat($(this).text()) === maxVal) {
                            $(this).css('font-weight', 'bold');
                        }
                    });
                }
            });
        })
        .catch(error => console.log('Error fetching data:', error));
    }, 1000); // Delay of 1000 milliseconds (1 second)
}

function showImage(image_name) {
    openImageModalBySrc(`../training/training_models/training_model_images/${image_name}.png`);
}

function fill_data_report(){
    let data_report_type = document.querySelector(`input[name='data_radio_group']:checked`).value;

    if (data_report_type === 'data-summary'){
        fill_data_summary_grid();
    }
    else if (data_report_type === 'train-data') {
        fill_train_data_report();
    }
    else if (data_report_type === 'test-data') {
        fill_test_data_report();
    }
}

function fill_data_summary_grid(){

    let grid = document.getElementById(`data-grid`);

    let selectedIds= [];
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedIds.push(id);
    });

    setTimeout(() => {
        fetch(`/training/get_data_summary?trainHistoryIds=${selectedIds}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {

            let column_title = data.columns.map(col => col.replace('Total', '').replace('Train', '').replace('Test', ''))

            let headersHtml = column_title.map(col => `<th>${col}</th>`).join('\n');
            let colSpan = ((data.columns.length) - 1) / 3

            grid.innerHTML = `<table id='data-summary-table' class="display">
                                <thead>
                                    <tr>
                                        <th>level</th>
                                        <th colspan="${colSpan}">Total</th>
                                        <th colspan="${colSpan}">Train</th>
                                        <th colspan="${colSpan}">Test</th>
                                    </tr>
                                    <tr id='data-summary-tableHeaders'>
                                        ${headersHtml}
                                    </tr>
                                </thead>
                                <tbody id='data-summary-tableBody'></tbody>
                            </table>`;


            // Initialize the DataTable with new data
            $(`#data-summary-table`).DataTable({
                destroy: true,
                data: data.data,
                scrollX: true,
                scrollY: 400,
                fixedColumns: {
                    leftColumns: 1,
                },
                columns: data.columns.map(col => ({ data: col })), // Map column names to data keys
                paging: false,
                ordering: false,
                searching: false,
                dom: 'Bfrtip',
                buttons: [{
                    extend: 'pdfHtml5',
                    title: 'Data Summary Table',
                    orientation: 'landscape',
                    pageSize: 'A4',
                    text: '<i class="fas fa-file-pdf"></i>',
                    exportOptions: {
                        // Export only visible columns
                        columns: ':visible',

                        modifier: {
                            search: 'applied',
                            order: 'applied',
                        }
                    }
                },
                    {
                        text: '<i class="fas fa-file-code"></i>',
                        action: function (e, dt, button, config) {
                            generateLatexCode(data.columns, data.data);
                        }
                    }
                ]
            });

        })
        .catch(error => console.log('Error fetching data:', error));
    }, 1000); // Delay of 1000 milliseconds (1 second)
}

function generateLatexCode(columns, data) {
    let latexCode = "\\begin{tabular}{|" + "c|".repeat(columns.length) + "} \\hline\n";

    latexCode += columns.join(' & ') + " \\\\ \\hline\n";

    data.forEach(row => {
        latexCode += columns.map(col => row[col] || "").join(' & ') + " \\\\ \\hline\n";
    });

    latexCode += "\\end{tabular}";

    console.log(latexCode);
    alert("LaTeX Code:\n\n" + latexCode);
}

function exportSelected(){
    let selectedIds= [];
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedIds.push(id);
    });

    fetch(`/training/export_data?trainIds=${selectedIds}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {



        hideSpinner(true);
    })
    .catch(error => {
        console.log('Error fetching data:', error)
        hideSpinner(true);
    });
}

function importSelected(){


    fetch(`/training/import_data`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {

        hideSpinner(true);
    })
    .catch(error => {
        console.log('Error fetching data:', error)
        hideSpinner(true);
    });

}

function fill_train_data_report(){

    let grid = document.getElementById(`data-grid`);

    let selectedIds= [];
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedIds.push(id);
    });

    setTimeout(() => {

        showSpinner();

        fetch(`/training/get_train_data_report?trainHistoryIds=${selectedIds}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {

            let column_title = data.columns.map(col => col.replace('Total', '').replace('Train', '').replace('Test', ''))

            let headersHtml = column_title.map(col => `<th>${col}</th>`).join('\n');

            grid.innerHTML = `<table id='data-summary-table' class="display">
                                <thead>
                                    <tr id='data-summary-tableHeaders'>
                                        ${headersHtml}
                                    </tr>
                                </thead>
                                <tbody id='data-summary-tableBody'></tbody>
                            </table>`;

            data.data = data.data.map(row => { for (let key in row) { if (row[key] === null) { row[key] = 'null'; } } return row; });

            $(`#data-summary-table`).DataTable({
                destroy: true,
                data: data.data,
                scrollX: true,
                scrollY: 400,
                fixedColumns: {
                    leftColumns: 1,
                },
                columns: data.columns.map(col => ({ data: col })), // Map column names to data keys
                paging: true,
                ordering: true,
                searching: true,
                dom: 'Bfrtip',
                buttons: [
                    {
                        extend: 'pdfHtml5',
                        title: 'Data Summary Table',
                        orientation: 'landscape',
                        pageSize: 'A4',
                        text: '<i class="fas fa-file-pdf"></i>',
                        exportOptions: {
                            columns: ':visible'
                        }
                    },
                    {
                        text: '<i class="fas fa-file-code"></i>',
                        action: function (e, dt, button, config) {
                            generateLatexCode(data.columns, data.data);
                        }
                    }
                ]
            });
            hideSpinner(true);
        })
        .catch(error => {
            console.log('Error fetching data:', error)
            hideSpinner(true);
        });
    }, 1000); // Delay of 1000 milliseconds (1 second)
}

function fill_validation_data_report(){

    let grid = document.getElementById(`data-grid`);

    let selectedIds= [];
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedIds.push(id);
    });

    setTimeout(() => {

        showSpinner();

        fetch(`/training/get_validation_data_report?trainHistoryIds=${selectedIds}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {

            let column_title = data.columns.map(col => col.replace('Total', '').replace('Train', '').replace('Test', ''))

            let headersHtml = column_title.map(col => `<th>${col}</th>`).join('\n');

            grid.innerHTML = `<table id='data-summary-table' class="display">
                                <thead>
                                    <tr id='data-summary-tableHeaders'>
                                        ${headersHtml}
                                    </tr>
                                </thead>
                                <tbody id='data-summary-tableBody'></tbody>
                            </table>`;

            data.data = data.data.map(row => { for (let key in row) { if (row[key] === null) { row[key] = 'null'; } } return row; });

            $(`#data-summary-table`).DataTable({
                destroy: true,
                data: data.data,
                scrollX: true,
                scrollY: 400,
                fixedColumns: {
                    leftColumns: 1,
                },
                columns: data.columns.map(col => ({ data: col })), // Map column names to data keys
                paging: true,
                ordering: true,
                searching: true,
                dom: 'Bfrtip',
                buttons: [
                    {
                        extend: 'pdfHtml5',
                        title: 'Data Summary Table',
                        orientation: 'landscape',
                        pageSize: 'A4',
                        text: '<i class="fas fa-file-pdf"></i>',
                        exportOptions: {
                            columns: ':visible'
                        }
                    },
                    {
                        text: '<i class="fas fa-file-code"></i>',
                        action: function (e, dt, button, config) {
                            generateLatexCode(data.columns, data.data);
                        }
                    }
                ]
            });
            hideSpinner(true);
        })
        .catch(error => {
            console.log('Error fetching data:', error)
            hideSpinner(true);
        });
    }, 1000); // Delay of 1000 milliseconds (1 second)
}

function fill_test_data_report(){

    let grid = document.getElementById(`data-grid`);

    let selectedIds= [];
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedIds.push(id);
    });

    setTimeout(() => {

        showSpinner();

        fetch(`/training/get_test_data_report?trainHistoryIds=${selectedIds}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {

            let column_title = data.columns.map(col => col.replace('Total', '').replace('Train', '').replace('Test', ''))

            let headersHtml = column_title.map(col => `<th>${col}</th>`).join('\n');

            data.data = data.data.map(row => { for (let key in row) { if (row[key] === null) { row[key] = 'null'; } } return row; });

            grid.innerHTML = `<table id='data-summary-table' class="display">
                                <thead>
                                    <tr id='data-summary-tableHeaders'>
                                        ${headersHtml}
                                    </tr>
                                </thead>
                                <tbody id='data-summary-tableBody'></tbody>
                            </table>`;


            // Initialize the DataTable with new data
            $(`#data-summary-table`).DataTable({
                destroy: true,
                data: data.data,
                scrollX: true,
                scrollY: 400,
                fixedColumns: {
                    leftColumns: 1,
                },
                columns: data.columns.map(col => ({ data: col })), // Map column names to data keys
                paging: true,
                ordering: true,
                searching: true,
                dom: 'Bfrtip',
                buttons: [
                    {
                        extend: 'pdfHtml5',
                        title: 'Data Summary Table',
                        orientation: 'landscape',
                        pageSize: 'A4',
                        text: '<i class="fas fa-file-pdf"></i>',
                        exportOptions: {
                            columns: ':visible'
                        }
                    },
                    {
                        text: '<i class="fas fa-file-code"></i>',
                        action: function (e, dt, button, config) {
                            generateLatexCode(data.columns, data.data);
                        }
                    }
                ]
            });
            hideSpinner(true);
        })
        .catch(error => {
            console.log('Error fetching data:', error)
            hideSpinner(true);

        });
    }, 1000); // Delay of 1000 milliseconds (1 second)
}

let currentGroupIndex = 0;
let currentSlideIndex = 0;

function openSlideshow(current_id) {
    currentGroupIndex = current_id;
    currentSlideIndex = 0;
    document.getElementById('slideshowModal').style.display = 'block';
    showSlides();
}

function closeSlideshow() {
    document.getElementById('slideshowModal').style.display = 'none';
}

function showSlides() {
    const container = document.getElementById('slideshowContainer');
    container.innerHTML = '';
    const images = imageGroups[currentGroupIndex];
    debugger;
    images.forEach((src, index) => {
        const slide = document.createElement('img');
        slide.src = `/training/training_history_plots/training_plots/${currentGroupIndex}/${src}`;
        slide.style.display = (index === currentSlideIndex) ? 'block' : 'none';
        container.appendChild(slide);
    });
}

function changeSlide(n) {
    const totalSlides = imageGroups[currentGroupIndex].length;
    currentSlideIndex = (currentSlideIndex + n + totalSlides) % totalSlides;
    showSlides();
}