self.addEventListener('install', event => {
    event.waitUntil(
        caches.open('wave-walker-v1').then(cache => {
            return cache.addAll([
                '/',
                '/static/p5.min.js',
                '/templates/index.html'
            ]);
        })
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(response => {
            return response || fetch(event.request);
        })
    );
});