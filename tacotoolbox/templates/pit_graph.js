/**
 * TACO PIT Structure Visualization
 * 
 * Renders interactive Position-Invariant Tree (PIT) diagrams using D3.js.
 * Auto-sizes to fit content with responsive configs for mobile/desktop.
 */

function getConfig() {
    const isMobile = window.innerWidth < 768;
    
    if (isMobile) {
        return {
            nodeRadius: 12,
            levelHeight: 60,
            nodeSpacingH: 45,
            nodeSpacingV: 50,
            startX: 120,
            startY: 35,
            maxChildrenFull: 4,
            leftMargin: 25,
            fontSize: { label: 10, type: 8, annotation: 12 },
            colors: { taco: '#FFB6C1', folder: '#90EE90', file: '#DDA0DD' }
        };
    }
    
    return {
        nodeRadius: 20,
        levelHeight: 100,
        nodeSpacingH: 90,
        nodeSpacingV: 80,
        startX: 200,
        startY: 50,
        maxChildrenFull: 4,
        leftMargin: 50,
        fontSize: { label: 14, type: 11, annotation: 16 },
        colors: { taco: '#FFB6C1', folder: '#90EE90', file: '#DDA0DD' }
    };
}

function renderPITGraph(pitSchema) {
    const container = d3.select('#pit-graph');
    container.selectAll('*').remove();
    
    if (!pitSchema || !pitSchema.root) {
        container.append('p')
            .style('color', '#999')
            .text('No structure data available');
        return;
    }
    
    const config = getConfig();
    const nodes = [];
    const edges = [];
    const texts = [];
    const colorMap = {
        'TACO': config.colors.taco,
        'FOLDER': config.colors.folder,
        'FILE': config.colors.file
    };
    
    // Create root node
    const root = {
        id: 'root',
        type: 'TACO',
        level: -1,
        x: config.startX,
        y: config.startY,
        label: 'TACO',
        color: colorMap['TACO']
    };
    nodes.push(root);
    
    // Create level 0 - handle different sample counts
    const level0Y = config.startY + config.levelHeight;
    const rootType = pitSchema.root.type;
    const nSamples = pitSchema.root.n;
    let sampleLeft;
    
    if (nSamples === 1) {
        // Single sample: centered, no ellipsis
        sampleLeft = {
            id: 'sample_0',
            type: rootType,
            level: 0,
            x: config.startX,
            y: level0Y,
            label: 'Sample 0',
            color: colorMap[rootType]
        };
        nodes.push(sampleLeft);
        edges.push({ fromX: root.x, fromY: root.y, toX: sampleLeft.x, toY: sampleLeft.y });
        
    } else if (nSamples === 2) {
        // Two samples: both visible, no ellipsis
        sampleLeft = {
            id: 'sample_0',
            type: rootType,
            level: 0,
            x: config.startX - config.nodeSpacingH,
            y: level0Y,
            label: 'Sample 0',
            color: colorMap[rootType]
        };
        nodes.push(sampleLeft);
        edges.push({ fromX: root.x, fromY: root.y, toX: sampleLeft.x, toY: sampleLeft.y });
        
        const sampleRight = {
            id: 'sample_1',
            type: rootType,
            level: 0,
            x: config.startX + config.nodeSpacingH,
            y: level0Y,
            label: 'Sample 1',
            color: colorMap[rootType]
        };
        nodes.push(sampleRight);
        edges.push({ fromX: root.x, fromY: root.y, toX: sampleRight.x, toY: sampleRight.y });
        
    } else {
        // Three or more samples: first, ellipsis, last
        sampleLeft = {
            id: 'sample_0',
            type: rootType,
            level: 0,
            x: config.startX - config.nodeSpacingH,
            y: level0Y,
            label: 'Sample 0',
            color: colorMap[rootType]
        };
        nodes.push(sampleLeft);
        edges.push({ fromX: root.x, fromY: root.y, toX: sampleLeft.x, toY: sampleLeft.y });
        
        texts.push({
            x: config.startX,
            y: level0Y + 5,
            text: '...',
            size: config.fontSize.annotation,
            weight: 'bold'
        });
        
        const sampleRight = {
            id: `sample_${nSamples - 1}`,
            type: rootType,
            level: 0,
            x: config.startX + config.nodeSpacingH,
            y: level0Y,
            label: `Sample ${nSamples - 1}`,
            color: colorMap[rootType]
        };
        nodes.push(sampleRight);
        edges.push({ fromX: root.x, fromY: root.y, toX: sampleRight.x, toY: sampleRight.y });
    }
    
    // Expand hierarchy recursively
    if (pitSchema.hierarchy) {
        let currentY = level0Y;
        let currentParent = sampleLeft;
        let currentDepth = 1;
        
        while (pitSchema.hierarchy[currentDepth.toString()]) {
            const patterns = pitSchema.hierarchy[currentDepth.toString()];
            if (!patterns || patterns.length === 0) break;
            
            const pattern = patterns[0];
            const types = pattern.type;
            const ids = pattern.id;
            currentY += config.nodeSpacingV;
            
            // Determine which children to show
            const numChildren = types.length;
            const indicesToShow = numChildren <= config.maxChildrenFull
                ? Array.from({ length: numChildren }, (_, i) => i)
                : [0, 1, numChildren - 1];
            const showEllipsis = numChildren > config.maxChildrenFull;
            const hiddenCount = numChildren - 3;
            
            // Create child nodes
            const numVisible = indicesToShow.length + (showEllipsis ? 1 : 0);
            const totalWidth = (numVisible - 1) * config.nodeSpacingH;
            let currentX = currentParent.x - (totalWidth / 2);
            let expandedChild = null;
            
            indicesToShow.forEach((i, idxPos) => {
                const childType = types[i];
                const childId = ids[i];
                const truncatedLabel = childId.length > 12 ? childId.substring(0, 12) + '...' : childId;
                
                const childNode = {
                    id: `child_d${currentDepth}_i${i}`,
                    type: childType,
                    level: currentDepth,
                    x: currentX,
                    y: currentY,
                    label: truncatedLabel,
                    fullLabel: childId,
                    color: colorMap[childType]
                };
                
                nodes.push(childNode);
                edges.push({
                    fromX: currentParent.x,
                    fromY: currentParent.y,
                    toX: childNode.x,
                    toY: childNode.y
                });
                
                // Expand first FOLDER found in visible children
                if (!expandedChild && childType === 'FOLDER') {
                    expandedChild = childNode;
                }
                
                currentX += config.nodeSpacingH;
                
                // Add ellipsis after second node
                if (showEllipsis && idxPos === 1) {
                    texts.push({
                        x: currentX,
                        y: currentY + 5,
                        text: `...(+${hiddenCount})`,
                        size: config.fontSize.annotation - 2,
                        weight: 'bold'
                    });
                    currentX += config.nodeSpacingH;
                }
            });
            
            // Add collapsed indicators for sibling folders
            if (expandedChild) {
                indicesToShow.slice(1).forEach(i => {
                    if (types[i] === 'FOLDER') {
                        const siblingNode = nodes.find(n => n.id === `child_d${currentDepth}_i${i}`);
                        if (siblingNode) {
                            texts.push({
                                x: siblingNode.x,
                                y: currentY + config.nodeSpacingV / 2,
                                text: '...',
                                size: config.fontSize.annotation,
                                weight: 'normal'
                            });
                        }
                    }
                });
            }
            
            if (!expandedChild) break;
            currentParent = expandedChild;
            currentDepth++;
        }
    }
    
    // Adjust positions to prevent left edge cutoff
    if (nodes.length > 0) {
        const minX = Math.min(...nodes.map(n => n.x));
        if (minX < config.leftMargin) {
            const offset = config.leftMargin - minX;
            nodes.forEach(node => node.x += offset);
            edges.forEach(edge => {
                edge.fromX += offset;
                edge.toX += offset;
            });
            texts.forEach(text => text.x += offset);
        }
    }
    
    // Calculate SVG dimensions
    const allX = [...nodes.map(n => n.x), ...texts.map(t => t.x)];
    const allY = [...nodes.map(n => n.y), ...texts.map(t => t.y)];
    const maxX = Math.max(...allX);
    const maxY = Math.max(...allY);
    const width = maxX + config.leftMargin + 30;
    const height = maxY + 60;
    
    // Create SVG
    const svg = container
        .append('svg')
        .attr('class', 'pit-graph-svg')
        .attr('width', '100%')
        .attr('height', '100%')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .attr('preserveAspectRatio', 'xMidYMid meet')
        .style('max-height', Math.min(height, 700) + 'px')
        .style('display', 'block')
        .style('margin', '0 auto');
    
    // Add arrow marker
    svg.append('defs')
        .append('marker')
        .attr('id', 'arrowhead')
        .attr('markerWidth', 10)
        .attr('markerHeight', 10)
        .attr('refX', 9)
        .attr('refY', 3)
        .attr('orient', 'auto')
        .append('polygon')
        .attr('points', '0 0, 10 3, 0 6')
        .attr('fill', '#999');
    
    // Render edges
    svg.selectAll('.edge')
        .data(edges)
        .enter()
        .append('line')
        .attr('class', 'edge')
        .attr('x1', d => d.fromX)
        .attr('y1', d => d.fromY)
        .attr('x2', d => {
            const dx = d.toX - d.fromX;
            const dy = d.toY - d.fromY;
            const length = Math.sqrt(dx * dx + dy * dy);
            return length > 0 ? d.fromX + dx * ((length - config.nodeRadius) / length) : d.toX;
        })
        .attr('y2', d => {
            const dx = d.toX - d.fromX;
            const dy = d.toY - d.fromY;
            const length = Math.sqrt(dx * dx + dy * dy);
            return length > 0 ? d.fromY + dy * ((length - config.nodeRadius) / length) : d.toY;
        })
        .attr('stroke', '#999')
        .attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#arrowhead)');
    
    // Render nodes
    const nodeGroup = svg.selectAll('.node')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', d => `translate(${d.x}, ${d.y})`);
    
    nodeGroup.append('circle')
        .attr('class', 'node-circle')
        .attr('r', config.nodeRadius)
        .attr('fill', d => d.color)
        .attr('stroke', '#666')
        .attr('stroke-width', 2);
    
    nodeGroup.append('title')
        .text(d => `${d.fullLabel || d.label} (${d.type})`);
    
    nodeGroup.append('text')
        .attr('class', 'node-label')
        .attr('text-anchor', 'middle')
        .attr('y', config.nodeRadius + 18)
        .style('font-size', config.fontSize.label + 'px')
        .style('font-weight', '600')
        .style('fill', '#333')
        .style('pointer-events', 'none')
        .text(d => d.label);
    
    nodeGroup.append('text')
        .attr('class', 'node-type-label')
        .attr('text-anchor', 'middle')
        .attr('y', config.nodeRadius + 32)
        .style('font-size', config.fontSize.type + 'px')
        .style('fill', '#666')
        .style('pointer-events', 'none')
        .text(d => d.type);
    
    // Render text annotations
    svg.selectAll('.annotation')
        .data(texts)
        .enter()
        .append('text')
        .attr('class', 'annotation')
        .attr('x', d => d.x)
        .attr('y', d => d.y)
        .attr('text-anchor', 'middle')
        .style('font-size', d => `${d.size}px`)
        .style('font-weight', d => d.weight || 'normal')
        .style('fill', '#666')
        .style('pointer-events', 'none')
        .text(d => d.text);
}

// Initialize and handle resize
document.addEventListener('DOMContentLoaded', function() {
    if (typeof pitSchemaData === 'undefined') {
        console.error('pitSchemaData not found - cannot render PIT graph');
        return;
    }
    
    renderPITGraph(pitSchemaData);
    
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            renderPITGraph(pitSchemaData);
        }, 250);
    });
});