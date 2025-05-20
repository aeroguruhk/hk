document.addEventListener('DOMContentLoaded', function() {
    const contactForm = document.querySelector('.contact-form form');
    const videoGrid = document.querySelector('.video-grid');
    
    // Gallery lightbox functionality
    const galleryItems = document.querySelectorAll('.gallery-item');
    const lightbox = document.querySelector('.lightbox');
    const lightboxImage = document.querySelector('.lightbox-image');
    const lightboxCaption = document.querySelector('.lightbox-caption');
    const lightboxClose = document.querySelector('.lightbox-close');
    const lightboxPrev = document.querySelector('.lightbox-prev');
    const lightboxNext = document.querySelector('.lightbox-next');
    
    let currentImageIndex = 0;
    
    // Open lightbox
    if (galleryItems.length > 0) {
        galleryItems.forEach((item, index) => {
            item.addEventListener('click', () => {
                const img = item.querySelector('img');
                const caption = item.querySelector('.gallery-caption');
                
                currentImageIndex = index;
                updateLightboxContent(img.src, caption.textContent);
                lightbox.classList.add('active');
                document.body.style.overflow = 'hidden'; // Prevent scrolling when lightbox is open
            });
        });
        
        // Update lightbox content
        function updateLightboxContent(src, caption) {
            lightboxImage.src = src;
            lightboxCaption.textContent = caption;
        }
        
        // Navigate to previous image
        lightboxPrev.addEventListener('click', (e) => {
            e.stopPropagation();
            currentImageIndex = (currentImageIndex - 1 + galleryItems.length) % galleryItems.length;
            const img = galleryItems[currentImageIndex].querySelector('img');
            const caption = galleryItems[currentImageIndex].querySelector('.gallery-caption');
            updateLightboxContent(img.src, caption.textContent);
        });
        
        // Navigate to next image
        lightboxNext.addEventListener('click', (e) => {
            e.stopPropagation();
            currentImageIndex = (currentImageIndex + 1) % galleryItems.length;
            const img = galleryItems[currentImageIndex].querySelector('img');
            const caption = galleryItems[currentImageIndex].querySelector('.gallery-caption');
            updateLightboxContent(img.src, caption.textContent);
        });
        
        // Close lightbox
        lightboxClose.addEventListener('click', (e) => {
            e.stopPropagation();
            lightbox.classList.remove('active');
            document.body.style.overflow = ''; // Re-enable scrolling
        });
        
        // Close lightbox when clicking outside the image
        lightbox.addEventListener('click', (e) => {
            if (e.target === lightbox) {
                lightbox.classList.remove('active');
                document.body.style.overflow = ''; // Re-enable scrolling
            }
        });
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (!lightbox.classList.contains('active')) return;
            
            if (e.key === 'Escape') {
                lightbox.classList.remove('active');
                document.body.style.overflow = ''; // Re-enable scrolling
            } else if (e.key === 'ArrowLeft') {
                lightboxPrev.click();
            } else if (e.key === 'ArrowRight') {
                lightboxNext.click();
            }
        });
    }
    
    // Handle contact form submission
    if (contactForm) {
        const submitButton = contactForm.querySelector('[type="submit"]');
        const originalButtonText = submitButton.textContent;
        
        contactForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Simple validation
            const name = contactForm.querySelector('[name="name"]')?.value.trim();
            const email = contactForm.querySelector('[name="email"]')?.value.trim();
            const message = contactForm.querySelector('[name="message"]')?.value.trim();
            
            if (!name || !email || !message) {
                alert('Please fill in all fields');
                return;
            }
            
            // Email validation
            if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
                alert('Please enter a valid email address');
                return;
            }
            
            try {
                // Update button state
                submitButton.textContent = 'Sending...';
                submitButton.disabled = true;
                
                const formData = new FormData(this);
                const response = await fetch(this.action, {
                    method: 'POST',
                    body: formData,
                    headers: {
                        'Accept': 'application/json'
                    }
                });
                
                if (response.ok) {
                    // Clear form
                    contactForm.reset();
                    alert('Thank you for your message! We will get back to you soon.');
                } else {
                    throw new Error('Form submission failed');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Sorry, there was an error sending your message. Please try again later.');
            } finally {
                // Reset button state
                submitButton.textContent = originalButtonText;
                submitButton.disabled = false;
            }
        });
    }

    // Load and display videos from links.txt
    if (videoGrid) {
        const lessonList = document.getElementById('lesson-list');
        let videos = [];

        // Add loading indicator
        videoGrid.innerHTML = '<div class="loading">Loading videos...</div>';

        // Set up AbortController for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

        fetch('links.txt', { signal: controller.signal })
            .then(response => {
                clearTimeout(timeoutId);
                if (!response.ok) {
                    throw new Error('Failed to fetch video links');
                }
                return response.text();
            })
            .then(data => {
                videos = data.split('\n')
                    .filter(line => line.trim() !== '')
                    .map(line => {
                        try {
                            const parts = line.split(', ').map(item => item.trim());
                            if (parts.length < 2) {
                                throw new Error('Invalid line format');
                            }
                            
                            const title = parts[0];
                            const youtubeUrl = parts[1];
                            const pdfUrl = parts[2] && parts[2].toLowerCase() !== 'none' ? parts[2] : null;
                            let videoId = '';
                            
                            // Extract video ID from different YouTube URL formats
                            try {
                                const url = youtubeUrl.trim();
                                if (url.includes('youtu.be/')) {
                                    videoId = url.split('youtu.be/')[1].split(/[?&]/)[0];
                                } else if (url.includes('youtube.com/watch')) {
                                    const urlObj = new URL(url);
                                    videoId = urlObj.searchParams.get('v');
                                } else if (url.includes('youtube.com/embed/')) {
                                    videoId = url.split('embed/')[1].split(/[?&]/)[0];
                                } else if (url.includes('youtube.com/v/')) {
                                    videoId = url.split('v/')[1].split(/[?&]/)[0];
                                } else if (url.match(/^[a-zA-Z0-9_-]{11}$/)) {
                                    // If it's just a video ID
                                    videoId = url;
                                }
                                
                                if (!videoId) {
                                    throw new Error(`Could not extract video ID from URL: ${url}`);
                                }
                            } catch (error) {
                                console.error('Error parsing YouTube URL:', youtubeUrl, error);
                                videoId = '';
                            }

                            return {
                                title,
                                videoId,
                                pdfUrl
                            };
                        } catch (error) {
                            console.error('Error parsing line:', line, error);
                            return null;
                        }
                    })
                    .filter(video => video !== null && video.videoId); // Filter out null entries and videos without ID

                // Populate sidebar with lesson names
                if (videos.length === 0) {
                    lessonList.innerHTML = '<li class="no-lessons">No videos available</li>';
                    videoGrid.innerHTML = '<p class="no-videos">No videos available.</p>';
                    return;
                }

                lessonList.innerHTML = videos.map((video, index) => 
                    `<li data-index="${index}" class="lesson-item${index === 0 ? ' active' : ''}">${video.title}</li>`
                ).join('');

                // Add click event listeners to lesson items
                const lessonItems = lessonList.querySelectorAll('.lesson-item');
                lessonItems.forEach(item => {
                    item.addEventListener('click', () => {
                        // Remove active class from all items
                        lessonItems.forEach(li => li.classList.remove('active'));
                        // Add active class to clicked item
                        item.classList.add('active');

                        const index = parseInt(item.dataset.index);
                        const video = videos[index];

                        // Display selected video with responsive container
                        videoGrid.innerHTML = `
                            <div class="video-wrapper">
                                <div class="video-title">
                                    <h3>${video.title}</h3>
                                </div>
                                <div class="video-responsive">
                                    <iframe
                                        src="https://www.youtube.com/embed/${video.videoId}"
                                        frameborder="0"
                                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                        allowfullscreen>
                                    </iframe>
                                </div>
                                ${video.pdfUrl ? `
                                <div class="pdf-download">
                                    <a href="${video.pdfUrl}" target="_blank" class="pdf-button">
                                        <span class="pdf-icon">📄</span>
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
                clearTimeout(timeoutId);
                console.error('Error loading videos:', error);
                videoGrid.innerHTML = `
                    <div class="error-message">
                        <p>Error loading videos. Please try again later.</p>
                        ${error.message ? `<p class="error-details">${error.message}</p>` : ''}
                    </div>
                `;
                lessonList.innerHTML = '<li class="error-item">Failed to load lessons</li>';
            });
    }
});