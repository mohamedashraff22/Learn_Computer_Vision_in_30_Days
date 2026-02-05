const video = document.getElementById("video");

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("./models"), // For Face Detection
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"), // For Face Recognition
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"), // For Face Landmarks
  // --- NEW MODELS ---
  faceapi.nets.faceExpressionNet.loadFromUri("./models"), // For Happy, Sad, Angry
  faceapi.nets.ageGenderNet.loadFromUri("./models")      // For Age and Gender
]).then(startWebcam);

function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({
      video: true,
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((error) => {
      console.error(error);
    });
}

function getLabeledFaceDescriptions() {
  const labels = ["mohamed", "yassin"];

  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      
      // Start checking from image 1
      let i = 1;
      
      // Loop forever until we hit a missing file
      while (true) {
        let img = null;
        
        try {
          // 1. Try to fetch the PNG
          img = await faceapi.fetchImage(`./labels/${label}/${i}.png`);
        } catch (error) {
          // PNG failed, so let's try JPG
          try {
            img = await faceapi.fetchImage(`./labels/${label}/${i}.jpg`);
          } catch (error2) {
            // Both PNG and JPG failed. 
            // This means we reached the end of the folder (e.g., file 6 doesn't exist).
            break; // STOP THE LOOP
          }
        }

        // If we found an image, detect the face
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detections) {
          descriptions.push(detections.descriptor);
          console.log(`Loaded ${label}/${i}`); // Debug message to see progress
        } else {
          console.warn(`Face not found in ${label}/${i}`);
        }

        // Move to the next image number (1 -> 2 -> 3...)
        i++;
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

video.addEventListener("play", async () => {
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceDescriptors()
      // --- NEW CHAINING ---
      .withFaceExpressions()  // Detect emotions
      .withAgeAndGender();    // Detect Age & Gender

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    // ... (Your existing face matching logic here) ...
    
    // Calculate match results
    const results = resizedDetections.map((d) => {
      return faceMatcher.findBestMatch(d.descriptor);
    });

    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const detection = resizedDetections[i]; // Get the full detection object

      // --- CUSTOM LABEL CREATION ---
      // We combine Name + Age + Gender into one string
      const { age, gender, genderProbability } = detection;
      const roundedAge = Math.round(age);
      
      // Example Label: "Mohamed (24 years, Male)"
      const labelText = `${result.toString()} (${roundedAge} yrs, ${gender})`;

      const drawBox = new faceapi.draw.DrawBox(box, {
        label: labelText,
      });
      drawBox.draw(canvas);
      
      // --- DRAW EMOTIONS ---
      // This draws the confidence bars (Happy: 90%, Sad: 2%)
      faceapi.draw.drawFaceExpressions(canvas, resizedDetections[i]);
    });
  }, 100);
});