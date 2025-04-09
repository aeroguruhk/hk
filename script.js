document.addEventListener('DOMContentLoaded', function() {
    const contactForm = document.querySelector('.contact-form form');
    const videoGrid = document.querySelector('.video-grid');
    
    // Handle contact form submission
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form values
            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const subject = document.getElementById('subject').value;
            const message = document.getElementById('message').value;
            
            // Here you can add code to handle the form submission
            // For example, sending data to a server
            console.log('Form submitted:', { name, email, subject, message });
            
            // Clear form
            contactForm.reset();
            alert('Thank you for your message! We will get back to you soon.');
        });
    }

    // Load and display videos from links.txt
    if (videoGrid) {
        const lessonList = document.getElementById('lesson-list');
        let videos = [];

        fetch('links.txt')
            .then(response => response.text())
            .then(data => {
                videos = data.split('\n')
                    .filter(line => line.trim() !== '')
                    .map(line => {
                        const [title, youtubeUrl, pdfUrl] = line.split(', ');
                        const videoId = youtubeUrl.split('v=')[1] || youtubeUrl.split('/').pop();
                        const cleanPdfUrl = pdfUrl ? pdfUrl.trim().toLowerCase() : '';
                        return { title, videoId, pdfUrl: (cleanPdfUrl === 'none' || cleanPdfUrl === 'no_pdf' || cleanPdfUrl === '') ? null : pdfUrl };
                    });

                // Populate sidebar with lesson names
                lessonList.innerHTML = videos.map((video, index) => `
                    <li data-index="${index}">${video.title}</li>
                `).join('');

                // Add click event listeners to lesson items
                const lessonItems = lessonList.querySelectorAll('li');
                lessonItems.forEach(item => {
                    item.addEventListener('click', () => {
                        // Remove active class from all items
                        lessonItems.forEach(li => li.classList.remove('active'));
                        // Add active class to clicked item
                        item.classList.add('active');

                        const index = parseInt(item.dataset.index);
                        const video = videos[index];

                        // Display selected video
                        videoGrid.innerHTML = `
                            <div class="video-placeholder">
                                <h4>${video.title}</h4>
                                <iframe width="100%" height="100%" 
                                    src="https://www.youtube.com/embed/${video.videoId}" 
                                    frameborder="0" 
                                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                                    allowfullscreen>
                                </iframe>
                                ${video.pdfUrl ? `
                                <div class="pdf-download">
                                    <a href="${video.pdfUrl}" target="_blank" class="pdf-button">
                                        <img src="pdf-icon.svg" alt="PDF" style="width: 20px; height: 20px; margin-right: 5px;">
                                        Download Study Material
                                    </a>
                                </div>` : ''}
                            </div>
                        `;
                    });
                });

                // Automatically select first video
                if (lessonItems.length > 0) {
                    lessonItems[0].click();
                }
            })
            .catch(error => {
                console.error('Error loading videos:', error);
                videoGrid.innerHTML = '<p>Error loading videos. Please try again later.</p>';
            });
    }
});