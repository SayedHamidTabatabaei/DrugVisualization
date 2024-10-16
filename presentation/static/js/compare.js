document.addEventListener('DOMContentLoaded', function() {

    get_history();
});

function get_history()
{
    const scenario = document.getElementById('scenarioSelect').value;
    const train_model = document.getElementById('trainModelSelect').value;

    showSpinner();

    fetch(`/training/get_history?start=0&length=10000&scenario=${scenario}&trainModel=${train_model}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {

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
                        render: function(data, type, row) {
                            const checkboxValue = `${row.id}|${row.name}`;

                            return '<label class="checkbox-label" style="width: 30px;">\n' +
                                '    <input type="checkbox" id="row-check" class="row-checkbox" onclick="compare_trainings()" value="' + checkboxValue + '">\n' +
                                '    <span class="checkbox-custom" style="margin-top: 0;"></span>' +
                                '</label>'
                        },
                        orderable: false,
                        searchable: false
                    },
                    {
                        render: function(data, type, row) {
                            return `<span title="${row.description}">${row.name}</span><br/><span>(${row.train_model})</span>`;
                        }
                    },
                    {
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

function filter_checked(){

    let isChecked = document.getElementById('filter-check').checked;

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

    let selectedItems = new Map();
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedItems.set(id, name);
    });

    const plotsGridContainer = document.getElementById('accuracyPlotsGridContainer');
    plotsGridContainer.innerHTML = '';

    selectedItems.forEach((value, key) => {

        const plotBox = document.createElement('div');
        plotBox.classList.add('plot-box');

        const plotImage = document.createElement('img');
        plotImage.src = `/training/training_history_plots/training_plots/${key}/accuracy_plot.png`;
        plotImage.alt = `Line Plot ${value}`;
        plotImage.onclick = () => openImageModal(plotImage);

        const caption = document.createElement('p');
        caption.classList.add('plot-caption');
        caption.textContent =value;

        // Append image to the plot box
        plotBox.appendChild(plotImage);
        plotBox.appendChild(caption);

        // Append plot box to the grid container
        plotsGridContainer.appendChild(plotBox);
    });
}

function fill_loss_plots(){

    let selectedItems = new Map();
    $('input.row-checkbox:checked').each(function() {

        const [id, name] = $(this).val().split('|'); // Get the value of the checked checkbox

        selectedItems.set(id, name);
    });

    const plotsGridContainer = document.getElementById('lossPlotsGridContainer');
    plotsGridContainer.innerHTML = '';

    selectedItems.forEach((value, key) => {

        const plotBox = document.createElement('div');
        plotBox.classList.add('plot-box');

        const plotImage = document.createElement('img');
        plotImage.src = `/training/training_history_plots/training_plots/${key}/loss_plot.png`;
        plotImage.alt = `Line Plot ${value}`;
        plotImage.onclick = () => openImageModal(plotImage);

        const caption = document.createElement('p');
        caption.classList.add('plot-caption');
        caption.textContent =value;

        // Append image to the plot box
        plotBox.appendChild(plotImage);
        plotBox.appendChild(caption);

        // Append plot box to the grid container
        plotsGridContainer.appendChild(plotBox);
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