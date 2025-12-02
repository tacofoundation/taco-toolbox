/**
 * TACO Dataset Map Visualization
 * 
 * Renders spatial extents using Leaflet with interactive popups.
 * Displays dataset coverage on an OpenStreetMap base layer with
 * color-coded partition rectangles and optional download links.
 */

const PARTITION_COLORS = [
    '#4CAF50',  // Green
    '#2196F3',  // Blue
    '#FF9800',  // Orange
    '#9C27B0',  // Purple
    '#F44336',  // Red
    '#00BCD4',  // Cyan
    '#FFEB3B',  // Yellow
    '#795548',  // Brown
    '#607D8B',  // Blue Grey
    '#E91E63'   // Pink
];

document.addEventListener('DOMContentLoaded', function() {
    if (typeof datasetExtents === 'undefined') {
        console.warn('datasetExtents not found - map will not render');
        return;
    }
    
    if (typeof L === 'undefined') {
        console.error('Leaflet library not loaded');
        return;
    }
    
    // Create base map
    const map = L.map('dataset-map', {
        zoomControl: true,
        attributionControl: true
    }).setView([0, 0], 2);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 18,
        minZoom: 2
    }).addTo(map);
    
    // Filter valid extents
    const validExtents = datasetExtents.filter(ext => 
        ext.spatial && Array.isArray(ext.spatial) && ext.spatial.length === 4
    );
    
    // Handle no data case
    if (validExtents.length === 0) {
        const noDataControl = L.control({ position: 'topright' });
        noDataControl.onAdd = function() {
            const div = L.DomUtil.create('div', 'map-info-box');
            div.innerHTML = '<strong>No spatial coverage data available</strong>';
            return div;
        };
        noDataControl.addTo(map);
        return;
    }
    
    // Check if download base URL is available
    const hasDownloads = typeof downloadBaseUrl !== 'undefined' && downloadBaseUrl !== null;
    
    // Render rectangles and collect bounds
    const bounds = [];
    
    validExtents.forEach((extent, index) => {
        const color = PARTITION_COLORS[index % PARTITION_COLORS.length];
        const [minLon, minLat, maxLon, maxLat] = extent.spatial;
        
        // Create rectangle
        const rectangle = L.rectangle(
            [[minLat, minLon], [maxLat, maxLon]], 
            {
                color: color,
                weight: 2,
                fillOpacity: 0.15,
                fillColor: color,
                className: 'partition-rect'
            }
        );
        
        // Build popup content
        const width = Math.abs(maxLon - minLon);
        const height = Math.abs(maxLat - minLat);
        const colorBox = `<span style="display:inline-block;width:12px;height:12px;background:${color};border-radius:2px;margin-right:8px;"></span>`;
        
        let popup = '<div class="map-popup">';
        popup += `<div class="map-popup-title">${colorBox}${extent.file || 'Unknown partition'}</div>`;
        
        if (extent.id) {
            popup += `<div class="map-popup-row">` +
                     `<span class="map-popup-label">Dataset:</span> ` +
                     `<span class="map-popup-value">${extent.id}</span>` +
                     `</div>`;
        }
        
        popup += `<div class="map-popup-row"><span class="map-popup-label">Spatial:</span></div>`;
        popup += `<div class="map-popup-row" style="margin-left:20px;">` +
                 `<span class="map-popup-value">[${minLon.toFixed(3)}, ${minLat.toFixed(3)}, ${maxLon.toFixed(3)}, ${maxLat.toFixed(3)}]</span>` +
                 `</div>`;
        popup += `<div class="map-popup-row" style="margin-left:20px;font-size:11px;color:#757575;">` +
                 `${width.toFixed(2)}&deg; &times; ${height.toFixed(2)}&deg;` +
                 `</div>`;
        
        if (extent.temporal && extent.temporal.length === 2) {
            const [start, end] = extent.temporal;
            popup += `<div class="map-popup-row"><span class="map-popup-label">Temporal:</span></div>`;
            popup += `<div class="map-popup-row" style="margin-left:20px;">` +
                     `<span class="map-popup-value">${start.substring(0, 10)} &rarr; ${end.substring(0, 10)}</span>` +
                     `</div>`;
        }
        
        // Add download button if base URL is available
        if (hasDownloads && extent.file) {
            const downloadUrl = downloadBaseUrl + extent.file;
            popup += `<div class="map-popup-download">` +
                     `<a href="${downloadUrl}" class="download-button" target="_blank" rel="noopener noreferrer">` +
                     `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">` +
                     `<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>` +
                     `<polyline points="7 10 12 15 17 10"></polyline>` +
                     `<line x1="12" y1="15" x2="12" y2="3"></line>` +
                     `</svg>` +
                     `Download File` +
                     `</a>` +
                     `</div>`;
        }
        
        popup += '</div>';
        
        rectangle.bindPopup(popup, {
            maxWidth: 350,
            className: 'partition-popup'
        });
        
        // Hover effects
        rectangle.on('mouseover', function() {
            this.setStyle({ weight: 3, fillOpacity: 0.3 });
        });
        
        rectangle.on('mouseout', function() {
            this.setStyle({ weight: 2, fillOpacity: 0.15 });
        });
        
        rectangle.addTo(map);
        bounds.push([[minLat, minLon], [maxLat, maxLon]]);
    });
    
    // Fit map to show all bounds
    if (bounds.length > 0) {
        const allBounds = bounds.reduce((acc, bound) => {
            return acc.extend(bound);
        }, L.latLngBounds(bounds[0]));
        
        map.fitBounds(allBounds, { 
            padding: [50, 50],
            maxZoom: 10
        });
    }
});