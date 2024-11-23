import React, { useRef, useState } from "react";

function App() {
  const videoRef = useRef(null);
  const [photo, setPhoto] = useState(null);
  const [result, setResult] = useState("");

  const startCamera = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => console.error("Error accessing camera:", err));
  };

  const takePicture = () => {
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    if (video) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const photoData = canvas.toDataURL("image/png");
      setPhoto(photoData);

      // Call backend function to determine the object (placeholder here)
      fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: photoData }),
      })
        .then((response) => response.json())
        .then((data) => setResult(data.result || "Unknown object"))
        .catch((err) => console.error("Error:", err));
    }
  };

  React.useEffect(() => {
    startCamera();
  }, []);

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Camera App</h1>
      <video ref={videoRef} autoPlay style={{ width: "100%", maxWidth: "500px" }} />
      <br />
      <button onClick={takePicture}>Take a Picture</button>
      {photo && (
        <div>
          <h3>Your Picture:</h3>
          <img src={photo} alt="Captured" style={{ width: "100%", maxWidth: "500px" }} />
          <h3>Detected Object:</h3>
          <p>{result || "Processing..."}</p>
        </div>
      )}
    </div>
  );
}

export default App;
