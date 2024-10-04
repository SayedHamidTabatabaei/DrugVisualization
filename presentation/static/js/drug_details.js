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

            visualizeMolecule(data.data[0].rdkit_3d)

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
    data.forEach(item => {
        const div = document.createElement('div');
        div.innerHTML = `
        <div><b>ID:</b> ${item.id}</div>
        <div><b>DrugBankId:</b> ${item.drugbank_id}</div>
        <div><b>Name:</b> ${item.drug_name}</div> 
        <div><b>Drug type:</b> ${item.drug_type}</div> 
        <div class="explanation" style="text-indent: -100px; padding-left: 100px;"><b>Description:</b> ${item.description}</div>  
        <div><b>Average mass:</b> ${item.average_mass}</div> 
        <div><b>Monoisotopic mass:</b> ${item.monoisotopic_mass}</div> 
        <div><b>State:</b> ${item.state}</div> 
        <div class="explanation" style="text-indent: -85px; padding-left: 85px;"><b>Indication:</b> ${item.indication}</div> 
        <div class="explanation" style="text-indent: -160px; padding-left: 160px;"><b>Pharmacodynamics:</b> ${item.pharmacodynamics}</div> 
        <div class="explanation" style="text-indent: -167px; padding-left: 167px;"><b>Mechanism of action:</b> ${item.mechanism_of_action}</div> 
        <div><b>Toxicity:</b> ${item.toxicity}</div> 
        <div><b>Metabolism:</b> ${item.metabolism}</div> 
        <div class="explanation" style="text-indent: -95px; padding-left: 95px;"><b>Absorption:</b> ${item.absorption}</div>
        <div><b>Half life:</b> ${item.half_life}</div>
        <div><b>Protein binding:</b> ${item.protein_binding}</div>
        <div class="explanation" style="text-indent: -165px; padding-left: 165px;"><b>Route of elimination:</b> ${item.route_of_elimination}</div>
        <div class="explanation" style="text-indent: -180px; padding-left: 180px;"><b>Volume of distribution:</b> ${item.volume_of_distribution}</div>
        <div><b>Clearance:</b> ${item.clearance}</div>
        <div class="explanation" style="text-indent: -205px; padding-left: 205px;"><b>Classification description:</b> ${item.classification_description}</div>
        <div><b>Classification direct parent:</b> ${item.classification_direct_parent}</div>
        <div><b>Classification kingdom:</b> ${item.classification_kingdom}</div>
        <div><b>Classification superclass:</b> ${item.classification_superclass}</div>
        <div><b>Classification class category:</b> ${item.classification_class_category}</div>
        <div><b>Classification subclass:</b> ${item.classification_subclass}</div>
        <div><b>Bioavailability:</b> ${item.bioavailability}</div>
        <div><b>Ghose filter:</b> ${item.ghose_filter}</div>
        <div><b>H bond acceptor count:</b> ${item.h_bond_acceptor_count}</div>
        <div><b>H bond donor count:</b> ${item.h_bond_donor_count}</div>
        <div><b>Log p:</b> ${item.log_p}</div>
        <div><b>log s:</b> ${item.log_s}</div>
        <div><b>Mddr like rule:</b> ${item.mddr_like_rule}</div>
        <div><b>Molecular formula:</b> ${item.molecular_formula}</div>
        <div><b>Molecular weight:</b> ${item.molecular_weight}</div>
        <div><b>Monoisotopic weight:</b> ${item.monoisotopic_weight}</div>
        <div><b>Number of rings:</b> ${item.number_of_rings}</div>
        <div><b>Physiological charge:</b> ${item.physiological_charge}</div>
        <div><b>Pka strongest acidic:</b> ${item.pka_strongest_acidic}</div>
        <div><b>Pka strongest basic:</b> ${item.pka_strongest_basic}</div>
        <div><b>Polar surface area:</b> ${item.polar_surface_area}</div>
        <div><b>Polarizability:</b> ${item.polarizability}</div>
        <div><b>Refractivity:</b> ${item.refractivity}</div>
        <div><b>Rotatable bond count:</b> ${item.rotatable_bond_count}</div>
        <div><b>Rule of five:</b> ${item.rule_of_five}</div>
        <div><b>Water solubility:</b> ${item.water_solubility}</div>`;
        container.appendChild(div);
    });

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
