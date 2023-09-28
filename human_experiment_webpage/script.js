let currentEpisode = 0;

let startTime = Date.now();

function getUrlParameter(name) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(name);
}
let concept_number=Number(getUrlParameter('id'));

let responses = [];
function arrayToBase64(array) {
    const string = JSON.stringify(array)
    return btoa(string);
}


// we load this from data.json
//let all_episodes=fetch('./data.json').then((response) => response.json())
let all_episodes={200: [[[true, {"color": "yellow", "shape": "circle", "size": "large"}], [false, {"color": "green", "shape": "triangle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "small"}], [true, {"color": "yellow", "shape": "rectangle", "size": "medium"}], [true, {"color": "yellow", "shape": "circle", "size": "small"}]], [[true, {"color": "green", "shape": "triangle", "size": "small"}], [true, {"color": "green", "shape": "circle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "medium"}], [false, {"color": "blue", "shape": "circle", "size": "large"}], [true, {"color": "green", "shape": "rectangle", "size": "medium"}], [true, {"color": "green", "shape": "circle", "size": "medium"}]], [[true, {"color": "blue", "shape": "circle", "size": "large"}], [true, {"color": "blue", "shape": "triangle", "size": "medium"}], [false, {"color": "green", "shape": "triangle", "size": "large"}], [true, {"color": "blue", "shape": "rectangle", "size": "medium"}], [true, {"color": "blue", "shape": "triangle", "size": "large"}]], [[true, {"color": "yellow", "shape": "triangle", "size": "medium"}], [true, {"color": "yellow", "shape": "circle", "size": "small"}], [true, {"color": "yellow", "shape": "circle", "size": "medium"}], [false, {"color": "green", "shape": "rectangle", "size": "large"}], [false, {"color": "green", "shape": "rectangle", "size": "medium"}]], [[false, {"color": "green", "shape": "triangle", "size": "large"}], [true, {"color": "yellow", "shape": "rectangle", "size": "small"}], [false, {"color": "green", "shape": "rectangle", "size": "small"}], [true, {"color": "yellow", "shape": "circle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "small"}]], [[false, {"color": "yellow", "shape": "triangle", "size": "large"}], [true, {"color": "blue", "shape": "rectangle", "size": "medium"}], [true, {"color": "blue", "shape": "circle", "size": "medium"}], [true, {"color": "blue", "shape": "rectangle", "size": "large"}], [false, {"color": "yellow", "shape": "circle", "size": "small"}], [true, {"color": "blue", "shape": "rectangle", "size": "small"}]], [[true, {"color": "yellow", "shape": "rectangle", "size": "large"}], [false, {"color": "blue", "shape": "rectangle", "size": "small"}], [true, {"color": "yellow", "shape": "circle", "size": "small"}], [false, {"color": "blue", "shape": "triangle", "size": "small"}], [true, {"color": "yellow", "shape": "rectangle", "size": "small"}]], [[true, {"color": "green", "shape": "rectangle", "size": "small"}], [true, {"color": "green", "shape": "triangle", "size": "medium"}], [true, {"color": "green", "shape": "circle", "size": "small"}], [false, {"color": "yellow", "shape": "triangle", "size": "large"}]], [[false, {"color": "blue", "shape": "triangle", "size": "large"}], [true, {"color": "yellow", "shape": "rectangle", "size": "medium"}], [true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "blue", "shape": "triangle", "size": "medium"}], [true, {"color": "yellow", "shape": "circle", "size": "medium"}]], [[true, {"color": "green", "shape": "triangle", "size": "small"}], [false, {"color": "yellow", "shape": "triangle", "size": "large"}], [true, {"color": "green", "shape": "circle", "size": "medium"}], [true, {"color": "green", "shape": "rectangle", "size": "large"}]], [[false, {"color": "yellow", "shape": "rectangle", "size": "large"}], [true, {"color": "blue", "shape": "rectangle", "size": "large"}], [true, {"color": "blue", "shape": "rectangle", "size": "small"}], [false, {"color": "yellow", "shape": "rectangle", "size": "small"}], [true, {"color": "blue", "shape": "triangle", "size": "large"}], [true, {"color": "blue", "shape": "rectangle", "size": "medium"}]], [[false, {"color": "blue", "shape": "circle", "size": "small"}], [true, {"color": "green", "shape": "circle", "size": "large"}], [true, {"color": "green", "shape": "rectangle", "size": "medium"}], [true, {"color": "green", "shape": "rectangle", "size": "small"}], [true, {"color": "green", "shape": "triangle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "large"}]], [[true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "blue", "shape": "rectangle", "size": "medium"}], [false, {"color": "blue", "shape": "circle", "size": "small"}], [true, {"color": "yellow", "shape": "rectangle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "medium"}]], [[true, {"color": "green", "shape": "rectangle", "size": "medium"}], [false, {"color": "yellow", "shape": "circle", "size": "small"}], [false, {"color": "yellow", "shape": "circle", "size": "medium"}], [true, {"color": "green", "shape": "rectangle", "size": "large"}], [true, {"color": "green", "shape": "circle", "size": "medium"}]], [[false, {"color": "yellow", "shape": "triangle", "size": "small"}], [true, {"color": "green", "shape": "circle", "size": "medium"}], [true, {"color": "green", "shape": "circle", "size": "small"}], [false, {"color": "yellow", "shape": "rectangle", "size": "large"}], [true, {"color": "green", "shape": "triangle", "size": "small"}]]], 201: [[[false, {"color": "green", "shape": "circle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "green", "shape": "triangle", "size": "small"}], [false, {"color": "green", "shape": "rectangle", "size": "medium"}], [false, {"color": "green", "shape": "circle", "size": "small"}]], [[false, {"color": "blue", "shape": "triangle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "small"}], [true, {"color": "green", "shape": "circle", "size": "medium"}], [true, {"color": "green", "shape": "circle", "size": "large"}], [false, {"color": "blue", "shape": "rectangle", "size": "medium"}], [false, {"color": "blue", "shape": "circle", "size": "medium"}]], [[false, {"color": "green", "shape": "circle", "size": "large"}], [false, {"color": "green", "shape": "triangle", "size": "medium"}], [true, {"color": "blue", "shape": "triangle", "size": "large"}], [false, {"color": "green", "shape": "rectangle", "size": "medium"}], [false, {"color": "green", "shape": "triangle", "size": "large"}]], [[false, {"color": "green", "shape": "triangle", "size": "medium"}], [false, {"color": "green", "shape": "circle", "size": "small"}], [false, {"color": "green", "shape": "circle", "size": "medium"}], [true, {"color": "yellow", "shape": "rectangle", "size": "large"}], [true, {"color": "yellow", "shape": "rectangle", "size": "medium"}]], [[true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "green", "shape": "rectangle", "size": "small"}], [true, {"color": "yellow", "shape": "rectangle", "size": "small"}], [false, {"color": "green", "shape": "circle", "size": "large"}], [false, {"color": "green", "shape": "triangle", "size": "small"}]], [[true, {"color": "blue", "shape": "triangle", "size": "large"}], [false, {"color": "yellow", "shape": "rectangle", "size": "medium"}], [false, {"color": "yellow", "shape": "circle", "size": "medium"}], [false, {"color": "yellow", "shape": "rectangle", "size": "large"}], [true, {"color": "blue", "shape": "circle", "size": "small"}], [false, {"color": "yellow", "shape": "rectangle", "size": "small"}]], [[false, {"color": "blue", "shape": "rectangle", "size": "large"}], [true, {"color": "yellow", "shape": "rectangle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "small"}], [true, {"color": "yellow", "shape": "triangle", "size": "small"}], [false, {"color": "blue", "shape": "rectangle", "size": "small"}]], [[false, {"color": "yellow", "shape": "rectangle", "size": "small"}], [false, {"color": "yellow", "shape": "triangle", "size": "medium"}], [false, {"color": "yellow", "shape": "circle", "size": "small"}], [true, {"color": "green", "shape": "triangle", "size": "large"}]], [[true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "blue", "shape": "rectangle", "size": "medium"}], [false, {"color": "blue", "shape": "triangle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "medium"}], [false, {"color": "blue", "shape": "circle", "size": "medium"}]], [[false, {"color": "yellow", "shape": "triangle", "size": "small"}], [true, {"color": "green", "shape": "triangle", "size": "large"}], [false, {"color": "yellow", "shape": "circle", "size": "medium"}], [false, {"color": "yellow", "shape": "rectangle", "size": "large"}]], [[true, {"color": "blue", "shape": "rectangle", "size": "large"}], [false, {"color": "yellow", "shape": "rectangle", "size": "large"}], [false, {"color": "yellow", "shape": "rectangle", "size": "small"}], [true, {"color": "blue", "shape": "rectangle", "size": "small"}], [false, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "yellow", "shape": "rectangle", "size": "medium"}]], [[true, {"color": "green", "shape": "circle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "large"}], [false, {"color": "blue", "shape": "rectangle", "size": "medium"}], [false, {"color": "blue", "shape": "rectangle", "size": "small"}], [false, {"color": "blue", "shape": "triangle", "size": "small"}], [true, {"color": "green", "shape": "circle", "size": "large"}]], [[false, {"color": "blue", "shape": "triangle", "size": "large"}], [true, {"color": "yellow", "shape": "rectangle", "size": "medium"}], [true, {"color": "yellow", "shape": "circle", "size": "small"}], [false, {"color": "blue", "shape": "rectangle", "size": "large"}], [false, {"color": "blue", "shape": "triangle", "size": "medium"}]], [[false, {"color": "yellow", "shape": "rectangle", "size": "medium"}], [true, {"color": "green", "shape": "circle", "size": "small"}], [true, {"color": "green", "shape": "circle", "size": "medium"}], [false, {"color": "yellow", "shape": "rectangle", "size": "large"}], [false, {"color": "yellow", "shape": "circle", "size": "medium"}]], [[true, {"color": "green", "shape": "triangle", "size": "small"}], [false, {"color": "yellow", "shape": "circle", "size": "medium"}], [false, {"color": "yellow", "shape": "circle", "size": "small"}], [true, {"color": "green", "shape": "rectangle", "size": "large"}], [false, {"color": "yellow", "shape": "triangle", "size": "small"}]]], 202: [[[true, {"color": "blue", "shape": "triangle", "size": "large"}], [false, {"color": "yellow", "shape": "rectangle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "small"}], [true, {"color": "green", "shape": "triangle", "size": "medium"}], [true, {"color": "blue", "shape": "triangle", "size": "small"}]], [[true, {"color": "yellow", "shape": "rectangle", "size": "small"}], [true, {"color": "blue", "shape": "rectangle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "medium"}], [false, {"color": "blue", "shape": "circle", "size": "large"}], [true, {"color": "green", "shape": "rectangle", "size": "medium"}], [true, {"color": "blue", "shape": "rectangle", "size": "medium"}]], [[true, {"color": "blue", "shape": "circle", "size": "large"}], [true, {"color": "yellow", "shape": "circle", "size": "medium"}], [false, {"color": "yellow", "shape": "rectangle", "size": "large"}], [true, {"color": "green", "shape": "circle", "size": "medium"}], [true, {"color": "yellow", "shape": "circle", "size": "large"}]], [[true, {"color": "yellow", "shape": "triangle", "size": "medium"}], [true, {"color": "blue", "shape": "triangle", "size": "small"}], [true, {"color": "blue", "shape": "triangle", "size": "medium"}], [false, {"color": "green", "shape": "rectangle", "size": "large"}], [false, {"color": "green", "shape": "rectangle", "size": "medium"}]], [[false, {"color": "yellow", "shape": "rectangle", "size": "large"}], [true, {"color": "green", "shape": "triangle", "size": "small"}], [false, {"color": "green", "shape": "rectangle", "size": "small"}], [true, {"color": "blue", "shape": "triangle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "small"}]], [[false, {"color": "yellow", "shape": "triangle", "size": "large"}], [true, {"color": "green", "shape": "circle", "size": "medium"}], [true, {"color": "blue", "shape": "circle", "size": "medium"}], [true, {"color": "green", "shape": "circle", "size": "large"}], [false, {"color": "blue", "shape": "triangle", "size": "small"}], [true, {"color": "green", "shape": "circle", "size": "small"}]], [[true, {"color": "green", "shape": "triangle", "size": "large"}], [false, {"color": "green", "shape": "circle", "size": "small"}], [true, {"color": "blue", "shape": "triangle", "size": "small"}], [false, {"color": "yellow", "shape": "circle", "size": "small"}], [true, {"color": "green", "shape": "triangle", "size": "small"}]], [[true, {"color": "green", "shape": "rectangle", "size": "small"}], [true, {"color": "yellow", "shape": "rectangle", "size": "medium"}], [true, {"color": "blue", "shape": "rectangle", "size": "small"}], [false, {"color": "yellow", "shape": "triangle", "size": "large"}]], [[false, {"color": "yellow", "shape": "circle", "size": "large"}], [true, {"color": "green", "shape": "triangle", "size": "medium"}], [true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "yellow", "shape": "circle", "size": "medium"}], [true, {"color": "blue", "shape": "triangle", "size": "medium"}]], [[true, {"color": "yellow", "shape": "rectangle", "size": "small"}], [false, {"color": "yellow", "shape": "triangle", "size": "large"}], [true, {"color": "blue", "shape": "rectangle", "size": "medium"}], [true, {"color": "green", "shape": "rectangle", "size": "large"}]], [[false, {"color": "green", "shape": "triangle", "size": "large"}], [true, {"color": "green", "shape": "circle", "size": "large"}], [true, {"color": "green", "shape": "circle", "size": "small"}], [false, {"color": "green", "shape": "triangle", "size": "small"}], [true, {"color": "yellow", "shape": "circle", "size": "large"}], [true, {"color": "green", "shape": "circle", "size": "medium"}]], [[false, {"color": "blue", "shape": "circle", "size": "small"}], [true, {"color": "blue", "shape": "rectangle", "size": "large"}], [true, {"color": "green", "shape": "rectangle", "size": "medium"}], [true, {"color": "green", "shape": "rectangle", "size": "small"}], [true, {"color": "yellow", "shape": "rectangle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "large"}]], [[true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "green", "shape": "circle", "size": "medium"}], [false, {"color": "blue", "shape": "circle", "size": "small"}], [true, {"color": "green", "shape": "triangle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "medium"}]], [[true, {"color": "green", "shape": "rectangle", "size": "medium"}], [false, {"color": "blue", "shape": "triangle", "size": "small"}], [false, {"color": "blue", "shape": "triangle", "size": "medium"}], [true, {"color": "green", "shape": "rectangle", "size": "large"}], [true, {"color": "blue", "shape": "rectangle", "size": "medium"}]], [[false, {"color": "yellow", "shape": "triangle", "size": "small"}], [true, {"color": "blue", "shape": "rectangle", "size": "medium"}], [true, {"color": "blue", "shape": "rectangle", "size": "small"}], [false, {"color": "green", "shape": "triangle", "size": "large"}], [true, {"color": "yellow", "shape": "rectangle", "size": "small"}]]], 203: [[[false, {"color": "blue", "shape": "rectangle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "yellow", "shape": "rectangle", "size": "small"}], [false, {"color": "green", "shape": "rectangle", "size": "medium"}], [false, {"color": "blue", "shape": "rectangle", "size": "small"}]], [[false, {"color": "yellow", "shape": "circle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "small"}], [true, {"color": "blue", "shape": "rectangle", "size": "medium"}], [true, {"color": "blue", "shape": "rectangle", "size": "large"}], [false, {"color": "green", "shape": "circle", "size": "medium"}], [false, {"color": "blue", "shape": "circle", "size": "medium"}]], [[false, {"color": "blue", "shape": "rectangle", "size": "large"}], [false, {"color": "yellow", "shape": "rectangle", "size": "medium"}], [true, {"color": "yellow", "shape": "circle", "size": "large"}], [false, {"color": "green", "shape": "rectangle", "size": "medium"}], [false, {"color": "yellow", "shape": "rectangle", "size": "large"}]], [[false, {"color": "yellow", "shape": "rectangle", "size": "medium"}], [false, {"color": "blue", "shape": "rectangle", "size": "small"}], [false, {"color": "blue", "shape": "rectangle", "size": "medium"}], [true, {"color": "green", "shape": "triangle", "size": "large"}], [true, {"color": "green", "shape": "triangle", "size": "medium"}]], [[true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "green", "shape": "rectangle", "size": "small"}], [true, {"color": "green", "shape": "triangle", "size": "small"}], [false, {"color": "blue", "shape": "rectangle", "size": "large"}], [false, {"color": "yellow", "shape": "rectangle", "size": "small"}]], [[true, {"color": "yellow", "shape": "circle", "size": "large"}], [false, {"color": "green", "shape": "triangle", "size": "medium"}], [false, {"color": "blue", "shape": "triangle", "size": "medium"}], [false, {"color": "green", "shape": "triangle", "size": "large"}], [true, {"color": "blue", "shape": "circle", "size": "small"}], [false, {"color": "green", "shape": "triangle", "size": "small"}]], [[false, {"color": "green", "shape": "circle", "size": "large"}], [true, {"color": "green", "shape": "triangle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "small"}], [true, {"color": "yellow", "shape": "triangle", "size": "small"}], [false, {"color": "green", "shape": "circle", "size": "small"}]], [[false, {"color": "green", "shape": "triangle", "size": "small"}], [false, {"color": "yellow", "shape": "triangle", "size": "medium"}], [false, {"color": "blue", "shape": "triangle", "size": "small"}], [true, {"color": "yellow", "shape": "rectangle", "size": "large"}]], [[true, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "green", "shape": "circle", "size": "medium"}], [false, {"color": "yellow", "shape": "circle", "size": "large"}], [true, {"color": "yellow", "shape": "triangle", "size": "medium"}], [false, {"color": "blue", "shape": "circle", "size": "medium"}]], [[false, {"color": "yellow", "shape": "triangle", "size": "small"}], [true, {"color": "yellow", "shape": "rectangle", "size": "large"}], [false, {"color": "blue", "shape": "triangle", "size": "medium"}], [false, {"color": "green", "shape": "triangle", "size": "large"}]], [[true, {"color": "green", "shape": "circle", "size": "large"}], [false, {"color": "green", "shape": "triangle", "size": "large"}], [false, {"color": "green", "shape": "triangle", "size": "small"}], [true, {"color": "green", "shape": "circle", "size": "small"}], [false, {"color": "yellow", "shape": "triangle", "size": "large"}], [false, {"color": "green", "shape": "triangle", "size": "medium"}]], [[true, {"color": "blue", "shape": "rectangle", "size": "small"}], [false, {"color": "blue", "shape": "circle", "size": "large"}], [false, {"color": "green", "shape": "circle", "size": "medium"}], [false, {"color": "green", "shape": "circle", "size": "small"}], [false, {"color": "yellow", "shape": "circle", "size": "small"}], [true, {"color": "blue", "shape": "rectangle", "size": "large"}]], [[false, {"color": "yellow", "shape": "circle", "size": "large"}], [true, {"color": "green", "shape": "triangle", "size": "medium"}], [true, {"color": "blue", "shape": "triangle", "size": "small"}], [false, {"color": "green", "shape": "circle", "size": "large"}], [false, {"color": "yellow", "shape": "circle", "size": "medium"}]], [[false, {"color": "green", "shape": "triangle", "size": "medium"}], [true, {"color": "blue", "shape": "rectangle", "size": "small"}], [true, {"color": "blue", "shape": "rectangle", "size": "medium"}], [false, {"color": "green", "shape": "triangle", "size": "large"}], [false, {"color": "blue", "shape": "triangle", "size": "medium"}]], [[true, {"color": "yellow", "shape": "rectangle", "size": "small"}], [false, {"color": "blue", "shape": "triangle", "size": "medium"}], [false, {"color": "blue", "shape": "triangle", "size": "small"}], [true, {"color": "green", "shape": "rectangle", "size": "large"}], [false, {"color": "yellow", "shape": "triangle", "size": "small"}]]]}



let episodes = all_episodes[concept_number]

function loadEpisode(episodeId) {
    document.getElementById('title').innerText = `Trial ${episodeId+1}`;
    if (episodeId>0){
	document.getElementById('title').innerText += `: Now you have feedback on what is wudsy`;
    } else {
	document.getElementById('title').innerText += `: Please read these instructions carefully`;
	}

    

    document.getElementById('nextButton').disabled = true;

    let training = episodes.slice(0, episodeId);
    let test = episodes[episodeId];

    responses.push(new Array(test.length).fill(null));
    
    const examplesDiv = document.getElementById('examples');
    examplesDiv.innerHTML = '';

    if (training.length==0){
	examplesDiv.innerHTML = 'You are going to attempt to learn the meaning of a new word in an alien language, which the aliens call "Wudsy." On each trial, you are going to see a collection of shapes at the bottom of the webpage, and your job is to select which ones you think are "Wudsy." Afterward, the aliens tell you which shapes are "Wudsy."<br><br>The meaning of the word Wudsy is the same during the whole experiment. However, it is possible that whether something is Wudsy depends on what other shapes it is in the context of. Wudsy may or may not correspond to an English word.<br><br>To start with, no one has given you any examples of what counts as "Wudsy." So just do your best below and pick which ones you think might belong to the concept called "Wudsy." Right after you do so, the aliens are going to label the Wudsy objects by drawing a black box around them, and then you are going to get another round of guessing which objects are "Wudsy."<br><br>You will go through 15 trials of guessing what counts as Wudsy. Remember that the meaning of Wudsy does not change during the experiment, but it might depend on the other shapes in the collection.<br><br>';
	
    } else {
	examplesDiv.innerHTML = 'You are going to attempt to learn the meaning of a new word in an alien language, which the aliens call "Wudsy." On each trial, you are going to see a collection of shapes at the bottom of the webpage, and your job is to select which ones you think are "Wudsy." Afterward, the aliens tell you which shapes are "Wudsy."<br><br>The meaning of the word Wudsy is the same during the whole experiment. However, it is possible that whether something is Wudsy depends on what other shapes it is in the context of. Wudsy may or may not correspond to an English word.<br><br>You will go through 15 trials of guessing what counts as Wudsy. Remember that the meaning of Wudsy does not change during the experiment, but it might depend on the other shapes in the collection.<br><br>';
	}

    for (let example of training) { 
        const exampleElement = document.createElement('p');
        exampleElement.innerText = `Here is an earlier example you have seen but with the wudsy objects indicated by a black border. Everything without a black border is not wudsy.\n\n `
	

        for (let training_object of example) {
	    let label = training_object[0];
	    let shape = training_object[1]["shape"];
	    let color = training_object[1]["color"];
	    let size = training_object[1]["size"];

	    let svgShape;
	    let shapeSize;
	    switch (shape) {
            case 'rectangle':
		svgShape = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
		shapeSize = `${size === 'small' ? 50 : size === 'medium' ? 75 : 100}`;
		svgShape.setAttribute('width', shapeSize);
		svgShape.setAttribute('height', shapeSize);
		break;
            case 'circle':
		svgShape = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
		shapeSize = `${size === 'small' ? 50 : size === 'medium' ? 75 : 100}`;
		svgShape.setAttribute('r', shapeSize / 2);
		break;
            case 'triangle':
		svgShape = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
		shapeSize = `${size === 'small' ? 50 : size === 'medium' ? 75 : 100}`;
		svgShape.setAttribute('points', `${shapeSize / 2},0 ${shapeSize},${shapeSize} 0,${shapeSize}`);
		break;
            default:
		console.error('Unknown shape:', shape);
	    }

	    svgShape.setAttribute('fill', color=="yellow"?"gold":color);

	    const svgContainer = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
	    svgContainer.setAttribute('width', '110');
	    svgContainer.setAttribute('height', '110');

	    // Center the shape within the SVG container
	    if (shape=="circle"){
		svgShape.setAttribute('transform', `translate(55, 55)`);
	    } else {
		svgShape.setAttribute('transform', `translate(${(100 - shapeSize) / 2+5}, ${(100 - shapeSize) / 2+5})`);
	    }
	    svgContainer.appendChild(svgShape);

	    // Add a black rectangle around the shape if the label is true
	    if (label) {
		const border = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
		border.setAttribute('width', 110);
		border.setAttribute('height', 110);
		border.setAttribute('fill', 'none');
		border.setAttribute('stroke', 'black');
		border.setAttribute('stroke-width', '3');
		svgContainer.appendChild(border);
	    }

	    const shapeContainer = document.createElement('div');
	    shapeContainer.style.display = 'inline-block';
	    shapeContainer.appendChild(svgContainer);
	    exampleElement.appendChild(shapeContainer);
	}


        examplesDiv.appendChild(exampleElement);
    }

    // new test examples section
    const testHeader = document.getElementById('probeheader');
    //const testHeaderText = document.createElement('p');
    testHeader.innerText = `(trial ${training.length+1}/${episodes.length}) Click yes on the objects that you think are Wudsy, and No on the other objects. Then click Next. `
    //testHeader.appendChild(testHeaderText)

    // new test examples section
    const testDiv = document.getElementById('probes');
    
    testDiv.innerHTML = '';
    testDiv.style.display = 'flex';
    testDiv.style.flexDirection = 'row';
    testDiv.style.alignItems = 'center';
    testDiv.style.justifyContent = 'left';

    function checkIfAllAnswered() {
	const allAnswered = responses[training.length].every(response => response !== null);
	document.getElementById('nextButton').disabled = !allAnswered;
    }

    for (let i = 0; i < test.length; i++) {
	let shape = test[i][1]["shape"];
	let color = test[i][1]["color"];
	let size = test[i][1]["size"];

	let svgShape;
	let shapeSize;
	switch (shape) {
        case 'rectangle':
	    svgShape = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
	    shapeSize = `${size === 'small' ? 50 : size === 'medium' ? 75 : 150}`;
	    svgShape.setAttribute('width', shapeSize);
	    svgShape.setAttribute('height', shapeSize);
	    break;
        case 'circle':
	    svgShape = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
	    shapeSize = `${size === 'small' ? 50 : size === 'medium' ? 75 : 100}`;
	    svgShape.setAttribute('r', shapeSize / 2);
	    break;
        case 'triangle':
	    svgShape = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
	    shapeSize = `${size === 'small' ? 50 : size === 'medium' ? 75 : 100}`;
	    svgShape.setAttribute('points', `${shapeSize / 2},0 ${shapeSize},${shapeSize} 0,${shapeSize}`);
	    break;
        default:
	    console.error('Unknown shape:', shape);
	}

	svgShape.setAttribute('fill', color=="yellow"?"gold":color);

	const svgContainer = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
	svgContainer.setAttribute('width', '110');
	svgContainer.setAttribute('height', '110');

	// Center the shape within the SVG container
	if (shape=="circle"){
	    svgShape.setAttribute('transform', `translate(55, 55)`);
	} else {
	    svgShape.setAttribute('transform', `translate(${(100 - shapeSize) / 2}, ${(100 - shapeSize) / 2})`);
	}
	svgContainer.appendChild(svgShape);

	const testElement = document.createElement('div');
	testElement.style.display = 'flex';
	testElement.style.flexDirection = 'column';
	testElement.style.alignItems = 'center';
	testElement.style.margin = '10px';

	shapeContainer=svgContainer
	testElement.appendChild(shapeContainer);

	const inputYes = document.createElement('input');
	inputYes.type = 'radio';
	inputYes.name = `test${i}`;
	inputYes.value = 'yes';
	inputYes.id = `test${i}yes`;
	inputYes.addEventListener('change', function() {
	    if(this.checked && this.value === 'yes')
		responses[training.length][i] = 1;
	    checkIfAllAnswered();
	});
	const labelYes = document.createElement('label');
	labelYes.for = `test${i}yes`;
	labelYes.innerText = 'Yes';

	const inputNo = document.createElement('input');
	inputNo.type = 'radio';
	inputNo.name = `test${i}`;
	inputNo.value = 'no';
	inputNo.id = `test${i}no`;
	inputNo.addEventListener('change', function() {
	    if(this.checked && this.value === 'no')
		responses[training.length][i] = 0;
	    checkIfAllAnswered();
	});
	const labelNo = document.createElement('label');
	labelNo.for = `test${i}no`;
	labelNo.innerText = 'No';

	testElement.appendChild(inputYes);
	testElement.appendChild(labelYes);
	testElement.appendChild(inputNo);
	testElement.appendChild(labelNo);

	testDiv.appendChild(testElement);
    }


    currentEpisode++;
}
    
function loadNextEpisode() {
    if (currentEpisode >= episodes.length) {
		// Calculate elapsed time in seconds
        let elapsedTime = Math.floor((Date.now() - startTime) / 1000);

        // Clear the screen
        document.body.innerHTML = '';

        // Convert responses to Base64
        const base64Responses = arrayToBase64(responses);

		// Create the URL
		const url = `https://docs.google.com/forms/d/e/1FAIpQLSeOUaUhMtmWgw2WhgTMa6SULk7z8xW8Ka9q2UXdpHGs_f125g/viewform?usp=pp_url&entry.436081205=${base64Responses}&entry.1082548960=${concept_number}&entry.629843327=${elapsedTime}`;

        // Display the message
        const message = document.createElement('p');
        message.textContent = `Thank you for participating! Please click the following link to complete the process (you will be redirected in 3 seconds): `;
        
        const link = document.createElement('a');
        link.href = url;
        link.textContent = 'Complete the Process';
        message.appendChild(link);
        
        document.body.appendChild(message);

	// Redirect after 3 seconds
        setTimeout(function() {
            window.location.href = url;
        }, 3000);

    } else {
        loadEpisode(currentEpisode);
	window.scrollTo(0,document.body.scrollHeight);
    }
}


loadEpisode(currentEpisode);
