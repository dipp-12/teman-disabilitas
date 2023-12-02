document.addEventListener('DOMContentLoaded', function() {
    // Connect with the socket server
    const socket = io.connect('https://vowel-detection-server-e3eec633c013.herokuapp.com/');    // Adjust the link with server

    const video = document.getElementById('video');
    const processedImage = document.getElementById('processed-image');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;

            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');

            setInterval(() => {
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg');
                socket.emit('image', imageData);
            }, 700); // Adjust the interval as needed
        })
        .catch(error => console.error('Error accessing webcam:', error));

    // Listen for the processed result from the server
    socket.on('processed_result', function(data) {
        processedImage.src = `data:image/jpeg;base64,${data}`;
    });
});
