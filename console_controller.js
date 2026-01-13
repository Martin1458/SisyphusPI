function generateConfig() {
	const dModel = document.getElementById("d_model").value || "128";
	const nHeads = document.getElementById("n_heads").value || "4";
	const weightDecay = document.getElementById("weight_decay").value || "1.0";
	const trainPct = document.getElementById("train_pct").value || "0.4";
	const steps = document.getElementById("steps").value || "100";

	const numSacrifices = document.getElementById("num_sacrifices").value || "5";
	const numWaves = document.getElementById("num_waves").value || "3";
	const minN = document.getElementById("min_n").value || "25";
	const maxN = document.getElementById("max_n").value || "30";
	const nStep = document.getElementById("n_step").value || "1";
	const weightDecays =
		document.getElementById("weight_decays").value || "1.0, 0.1, 0.01";
	const learningRates =
		document.getElementById("learning_rates").value ||
		"0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1";

	const projectName =
		document.getElementById("project_name").value || "default_project";
	const plotting = document.getElementById("plotting").checked ? "true" : "false";

	const ini = [
		"[model]",
		`D_MODEL = ${dModel}`,
		`N_HEADS = ${nHeads}`,
		`WEIGHT_DECAY = ${weightDecay}`,
		`TRAIN_PCT = ${trainPct}`,
		`STEPS = ${steps}`,
		"",
		"[training]",
		`NUM_OF_SACRIFICES = ${numSacrifices}`,
		`NUM_OF_WAVES = ${numWaves}`,
		`MIN_N = ${minN}`,
		`MAX_N = ${maxN}`,
		`N_STEP = ${nStep}`,
		`WEIGHT_DECAYS = ${weightDecays}`,
		`LEARNING_RATES = ${learningRates}`,
		"",
		"[project]",
		`PROJECT_NAME = ${projectName}`,
		"",
		"[plot]",
		`PLOTTING = ${plotting}`,
		"",
	].join("\n");

	document.getElementById("config-output").value = ini;
}

function downloadConfig() {
	const text = document.getElementById("config-output").value || "";
	if (!text.trim()) {
		generateConfig();
	}
	const finalText = document.getElementById("config-output").value;
	const blob = new Blob([finalText], { type: "text/plain" });
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = "config.ini";
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
}

async function saveConfigToExistingFile() {
	// Generate latest config text
	if (!document.getElementById("config-output").value.trim()) {
		generateConfig();
	}
	const finalText = document.getElementById("config-output").value;

	if (!("showSaveFilePicker" in window)) {
		alert(
			"Your browser does not support direct file saving. Use 'Download config.ini' and overwrite config.ini manually."
		);
		return;
	}

	try {
		const handle = await window.showSaveFilePicker({
			suggestedName: "config.ini",
			types: [
				{
					description: "INI files",
					accept: { "text/plain": [".ini"] },
				},
			],
		});
		const writable = await handle.createWritable();
		await writable.write(finalText);
		await writable.close();
		alert("config.ini saved.");
	} catch (err) {
		if (err && err.name !== "AbortError") {
			console.error("Error saving config.ini", err);
			alert("Failed to save config.ini. See console for details.");
		}
	}
}

window.addEventListener("DOMContentLoaded", generateConfig);

