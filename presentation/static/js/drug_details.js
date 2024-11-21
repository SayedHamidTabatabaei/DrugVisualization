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

// Show the first tab by default
document.addEventListener("DOMContentLoaded", function() {
    document.querySelector(".tab").click();
});

document.addEventListener('DOMContentLoaded', function() {
    const drugbank_id = document.getElementById('drugbank-id').textContent;

    fetch(`/drug/information/${encodeURIComponent(drugbank_id)}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            fillInformationContainer(data.data)

            visualizeMolecule(data.data.rdkit_3d)

            hideError();
        }
        else{
            showError("Error: DrugBank ID cannot be empty.");
        }
    })
    .catch(error => console.log('Error:', error));
});

function visualizeMolecule(rdkitMol) {

    const resultContainer = document.getElementById('result');

    if(!rdkitMol){
        const mol_container = document.getElementById('mol-container');
        mol_container.innerHTML = '';
        resultContainer.innerHTML = `<p>This drug hasn't had any visualization part!</p>`;
        resultContainer.display = "none";

        return
    }

    resultContainer.display = "block";

    resultContainer.innerHTML = '';

    let viewer = $3Dmol.createViewer('mol-container', {
        defaultcolors: $3Dmol.rasmolElementColors
    });

    viewer.addModel(rdkitMol, 'sdf');

    viewer.setStyle({}, {stick:{}});

    viewer.zoomTo();

    viewer.render();
}

function fillInformationContainer(data)
{
    const container = document.getElementById('data-container');
    container.innerHTML = '';
    const div = document.createElement('div');
    div.innerHTML = `
    <div><b>ID:</b> ${data.id}</div>
    <div><b>DrugBankId:</b> ${data.drugbank_id}</div>
    <div><b>Name:</b> ${data.drug_name}</div> 
    <div><b>Drug type:</b> ${data.drug_type}</div> 
    <div class="explanation" style="text-indent: -100px; padding-left: 100px;"><b>Description:</b> ${data.description}</div>  
    <div><b>Average mass:</b> ${data.average_mass}</div> 
    <div><b>Monoisotopic mass:</b> ${data.monoisotopic_mass}</div> 
    <div><b>State:</b> ${data.state}</div> 
    <div class="explanation" style="text-indent: -85px; padding-left: 85px;"><b>Indication:</b> ${data.indication}</div> 
    <div class="explanation" style="text-indent: -160px; padding-left: 160px;"><b>Pharmacodynamics:</b> ${data.pharmacodynamics}</div> 
    <div class="explanation" style="text-indent: -167px; padding-left: 167px;"><b>Mechanism of action:</b> ${data.mechanism_of_action}</div> 
    <div><b>Toxicity:</b> ${data.toxicity}</div> 
    <div><b>Metabolism:</b> ${data.metabolism}</div> 
    <div class="explanation" style="text-indent: -95px; padding-left: 95px;"><b>Absorption:</b> ${data.absorption}</div>
    <div><b>Half life:</b> ${data.half_life}</div>
    <div><b>Protein binding:</b> ${data.protein_binding}</div>
    <div class="explanation" style="text-indent: -165px; padding-left: 165px;"><b>Route of elimination:</b> ${data.route_of_elimination}</div>
    <div class="explanation" style="text-indent: -180px; padding-left: 180px;"><b>Volume of distribution:</b> ${data.volume_of_distribution}</div>
    <div><b>Clearance:</b> ${data.clearance}</div>
    <div class="explanation" style="text-indent: -205px; padding-left: 205px;"><b>Classification description:</b> ${data.classification_description}</div>
    <div class="explanation" style="text-indent: -205px; padding-left: 205px;"><b>Total Text:</b> ${data.total_text}</div>
    <div><b>Classification direct parent:</b> ${data.classification_direct_parent}</div>
    <div><b>Classification kingdom:</b> ${data.classification_kingdom}</div>
    <div><b>Classification superclass:</b> ${data.classification_superclass}</div>
    <div><b>Classification class category:</b> ${data.classification_class_category}</div>
    <div><b>Classification subclass:</b> ${data.classification_subclass}</div>
    <div><b>Bioavailability:</b> ${data.bioavailability}</div>
    <div><b>Ghose filter:</b> ${data.ghose_filter}</div>
    <div><b>H bond acceptor count:</b> ${data.h_bond_acceptor_count}</div>
    <div><b>H bond donor count:</b> ${data.h_bond_donor_count}</div>
    <div><b>Log p:</b> ${data.log_p}</div>
    <div><b>log s:</b> ${data.log_s}</div>
    <div><b>Mddr like rule:</b> ${data.mddr_like_rule}</div>
    <div><b>Molecular formula:</b> ${data.molecular_formula}</div>
    <div><b>Molecular weight:</b> ${data.molecular_weight}</div>
    <div><b>Monoisotopic weight:</b> ${data.monoisotopic_weight}</div>
    <div><b>Number of rings:</b> ${data.number_of_rings}</div>
    <div><b>Physiological charge:</b> ${data.physiological_charge}</div>
    <div><b>Pka strongest acidic:</b> ${data.pka_strongest_acidic}</div>
    <div><b>Pka strongest basic:</b> ${data.pka_strongest_basic}</div>
    <div><b>Polar surface area:</b> ${data.polar_surface_area}</div>
    <div><b>Polarizability:</b> ${data.polarizability}</div>
    <div><b>Refractivity:</b> ${data.refractivity}</div>
    <div><b>Rotatable bond count:</b> ${data.rotatable_bond_count}</div>
    <div><b>Rule of five:</b> ${data.rule_of_five}</div>
    <div><b>Water solubility:</b> ${data.water_solubility}</div>`;
    container.appendChild(div);
}

function showError(message) {
    const errorMessage = document.getElementById("error-message");
    errorMessage.textContent = message;
    errorMessage.style.display = "block";
}

function hideError() {
    const errorMessage = document.getElementById("error-message");
    errorMessage.textContent = '';
    errorMessage.style.display = "none";
}

function cleanElementSymbols(input) {
    return input.replace(/^\d+\((\w+)\)$/, '$1');
}

function fillAdjacency(){
    const drugbank_id = document.getElementById('drugbank-id').textContent;

    let grid = document.getElementById(`adjacency-grid`);

    setTimeout(() => {

        showSpinner();

        fetch(`/drug/adjacency_matrix/${drugbank_id}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {

            // let headersHtml = data.columns.map(col => `<th><span style="display: none">${col}</span><span>${cleanElementSymbols(col)}</span></th>`).join('\n');

            const tableHeaders = document.getElementById('adjacency-tableHeaders');
            tableHeaders.innerHTML = '';
            data.columns.forEach(col => {
                const th = document.createElement('th');
                th.innerHTML = `<span style="display: none">${col}</span><span>${cleanElementSymbols(col)}</span>`;
                tableHeaders.appendChild(th);
            });

            // Initialize the DataTable with new data
            $(`#adjacency-table`).DataTable({
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
                initComplete: function() {
                    const wrapper = document.getElementById('adjacency-table_wrapper');

                    if (wrapper) {
                        const scrollBody = wrapper
                            .querySelectorAll('.row')[1]
                            ?.querySelector('.col-sm-12')
                            ?.querySelector('.dataTables_scroll')
                            ?.querySelector('.dataTables_scrollBody');

                        if (scrollBody) {
                            const calculatedMaxWidth = calculateMaxWidth(data.columns.length);  // Define your calculateMaxWidth function
                            scrollBody.style.maxWidth = calculatedMaxWidth || '100%';
                        }
                    }
    }
            });
            hideSpinner(true);
        })
        .catch(error => {
            console.log('Error fetching data:', error)
            hideSpinner(true);

        });
    }, 1000); // Delay of 1000 milliseconds (1 second)

    function calculateMaxWidth(column_count) {
        return `${140 + ((column_count - 1) * 60)}px`;
    }
}

function fillEnzyme() {
    const drugbank_id = document.getElementById('drugbank-id').textContent;

    fetch(`/enzyme/enzymes/${encodeURIComponent(drugbank_id)}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            $('#enzymeTable').show();

            $('#enzymeTable').DataTable({
                data: data.data,
                destroy: true,
                columns: [
                    { data: 'id' },
                    { data: 'enzyme_code' },
                    { data: 'enzyme_name', className: 'forty-percent-width' },
                    { data: 'position' },
                    { data: 'organism' }
                ],
                paging: true,
                searching: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                language: {
                    search: "_INPUT_",
                    searchPlaceholder: "Search enzymes..."
                },
                initComplete: function() {
                    let table = document.getElementById('enzymeTable');
                    if (table) {
                        table.removeAttribute('style');
                    }
                }
            });

        } else {
            $('#enzymeTable').hide();
            console.log('Error: No data found.');
        }
    })
    .catch(error => console.log('Error:', error));
}

function fillTarget() {
    const drugbank_id = document.getElementById('drugbank-id').textContent;

    fetch(`/target/targets/${encodeURIComponent(drugbank_id)}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            $('#targetTable').show();

            $('#targetTable').DataTable({
                data: data.data,
                destroy: true,
                columns: [
                    { data: 'id' },
                    { data: 'target_code' },
                    { data: 'target_name', className: 'forty-percent-width' },
                    { data: 'position' },
                    { data: 'organism' }
                ],
                paging: true,
                searching: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                language: {
                    search: "_INPUT_",
                    searchPlaceholder: "Search targets..."
                },
                initComplete: function() {
                    let table = document.getElementById('targetTable');
                    if (table) {
                        table.removeAttribute('style');
                    }
                }
            });

        } else {
            $('#targetTable').hide();
            console.log('Error: No data found.');
        }
    })
    .catch(error => console.log('Error:', error));
}

function fillPathway() {
    const drugbank_id = document.getElementById('drugbank-id').textContent;

    fetch(`/pathway/pathways/${encodeURIComponent(drugbank_id)}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            $('#pathwayTable').show();

            $('#pathwayTable').DataTable({
                data: data.data,
                destroy: true,
                columns: [
                    { data: 'id', className: 'ten-percent-width' },
                    { data: 'pathway_code', className: 'thirty-percent-width' },
                    { data: 'pathway_name', className: 'sixty-percent-width' }
                ],
                paging: true,
                searching: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                language: {
                    search: "_INPUT_",
                    searchPlaceholder: "Search pathways..."
                },
                initComplete: function() {
                    let table = document.getElementById('pathwayTable');
                    if (table) {
                        table.removeAttribute('style');
                    }
                }
            });

        } else {
            $('#pathwayTable').hide();
            console.log('Error: No data found.');
        }
    })
    .catch(error => console.log('Error:', error));
}


function fillInteraction() {
    const drugbank_id = document.getElementById('drugbank-id').textContent;

    fetch(`/drug/interactions/${encodeURIComponent(drugbank_id)}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status) {
            $('#interactionTable').show();

            $('#interactionTable').DataTable({
                data: data.data,
                destroy: true,
                columns: [
                    { data: 'destination_drugbank_id', className: 'twenty-percent-width' },
                    { data: 'destination_drug_name', className: 'thirty-percent-width' },
                    { data: 'description', className: 'sixty-percent-width' }
                ],
                paging: true,
                searching: true,
                ordering: true,
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]],
                language: {
                    search: "_INPUT_",
                    searchPlaceholder: "Search Interactions..."
                },
                initComplete: function() {
                    let table = document.getElementById('interactionTable');
                    if (table) {
                        table.removeAttribute('style');
                    }
                }
            });

        } else {
            $('#interactionTable').hide();
            console.log('Error: No data found.');
        }
    })
    .catch(error => console.log('Error:', error));
}
