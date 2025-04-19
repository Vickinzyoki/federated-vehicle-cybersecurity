document.addEventListener('DOMContentLoaded', () => {
    const socket = io({
        transports: ['websocket', 'polling'],
        upgrade: true,
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000
    });

    let trainingProgress;
    let attackDistribution;
    const metrics = {
        accuracy: [],
        precision: [],
        recall: [],
        f1Score: []
    };

    function initializeCharts() {
        // Initialize Training Progress Chart
        const progressData = [{
            y: [],
            mode: 'lines+markers',
            name: 'Accuracy',
            line: { color: '#2ecc71' }
        }];
        
        const progressLayout = {
            margin: { t: 20, r: 20, b: 40, l: 40 },
            showlegend: false,
            xaxis: { title: 'Round' },
            yaxis: { title: 'Accuracy', range: [0, 1] },
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: '#ffffff'
        };
        
        Plotly.newPlot('training-progress', progressData, progressLayout);
        
        // Initialize Attack Distribution Chart
        const distributionData = [{
            values: [25, 25, 25, 25],
            labels: ['DoS', 'Fuzzing', 'Replay', 'Impersonation'],
            type: 'pie',
            marker: {
                colors: ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
            }
        }];
        
        const distributionLayout = {
            margin: { t: 20, r: 20, b: 20, l: 20 },
            showlegend: true,
            legend: { orientation: 'h', y: -0.2 }
        };
        
        Plotly.newPlot('attack-distribution', distributionData, distributionLayout);
    }

    function updateMetrics(data) {
        const metricsToUpdate = ['accuracy', 'precision', 'recall', 'f1-score'];
        metricsToUpdate.forEach(metric => {
            const value = data[metric.replace('-', '_')] || 0;
            const element = document.getElementById(metric);
            if (element) {
                element.textContent = value.toFixed(3);
                
                // Update trend
                metrics[metric.replace('-', '')].push(value);
                const trend = document.getElementById(`${metric}-trend`);
                if (trend && metrics[metric.replace('-', '')].length > 1) {
                    const lastTwo = metrics[metric.replace('-', '')].slice(-2);
                    const improvement = ((lastTwo[1] - lastTwo[0]) / lastTwo[0]) * 100;
                    trend.style.backgroundColor = improvement >= 0 ? '#2ecc71' : '#e74c3c';
                    trend.style.width = `${Math.min(Math.abs(improvement), 100)}%`;
                }
            }
        });
    }

    function updateTrainingProgress(data) {
        const update = {
            y: [[data.accuracy]],
            x: [[data.round]]
        };
        Plotly.extendTraces('training-progress', update, [0]);

        // Update layout to show proper range
        const layout = {
            margin: { t: 20, r: 20, b: 40, l: 40 },
            showlegend: false,
            xaxis: { title: 'Round', range: [0, Math.max(10, data.round + 1)] },
            yaxis: { title: 'Accuracy', range: [0, 1] },
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: '#ffffff'
        };
        Plotly.relayout('training-progress', layout);
    }

    function updateAttackDistribution(data) {
        const update = {
            values: [[data.dos, data.fuzzing, data.replay, data.impersonation]]
        };
        Plotly.update('attack-distribution', update);
    }

    function addDetection(detection) {
        const list = document.getElementById('detection-list');
        const empty = list.querySelector('.empty-state');
        if (empty) empty.remove();
        
        const item = document.createElement('div');
        item.className = 'detection-item';
        item.innerHTML = `
            <span class="detection-type">${detection.type}</span>
            <span class="detection-time">${new Date().toLocaleTimeString()}</span>
        `;
        
        list.insertBefore(item, list.firstChild);
        if (list.children.length > 10) {
            list.removeChild(list.lastChild);
        }
    }

    function updateClientStatus(clients) {
        console.log('Updating client status:', clients);
        const grid = document.querySelector('.client-grid');
        if (!grid) {
            console.error('Client grid not found!');
            return;
        }
        grid.innerHTML = '';
        
        if (!Array.isArray(clients)) {
            console.error('Invalid clients data:', clients);
            return;
        }
        
        clients.forEach(client => {
            const div = document.createElement('div');
            div.className = `client ${client.active ? 'active' : 'inactive'}`;
            div.innerHTML = `
                <div class="vehicle-id">Vehicle ${client.id}</div>
                <div class="status">${client.active ? 'Connected' : 'Offline'}</div>
            `;
            grid.appendChild(div);
        });
        
        // If no clients, show a message
        if (clients.length === 0) {
            const div = document.createElement('div');
            div.className = 'client inactive';
            div.innerHTML = `
                <div class="vehicle-id">No Vehicles</div>
                <div class="status">Waiting for connections...</div>
            `;
            grid.appendChild(div);
        }
    }

    // Socket event handlers
    socket.on('connect', () => {
        console.log('Connected to server');
        const indicator = document.getElementById('status-indicator');
        indicator.classList.add('online');
        indicator.textContent = 'Connected';
        document.getElementById('start-button').disabled = false;
        
        // Request initial state
        socket.emit('request_state');
    });

    socket.on('connect_error', (error) => {
        console.log('Connection error:', error);
        const indicator = document.getElementById('status-indicator');
        indicator.classList.remove('online');
        indicator.textContent = 'Connection Error';
        document.getElementById('start-button').disabled = true;
    });

    socket.on('reconnect', (attemptNumber) => {
        console.log('Reconnected after', attemptNumber, 'attempts');
        const indicator = document.getElementById('status-indicator');
        indicator.classList.add('online');
        indicator.textContent = 'Connected';
        document.getElementById('start-button').disabled = false;
    });

    socket.on('reconnect_error', (error) => {
        console.log('Reconnection error:', error);
        const indicator = document.getElementById('status-indicator');
        indicator.classList.remove('online');
        indicator.textContent = 'Reconnection Error';
        document.getElementById('start-button').disabled = true;
    });

    socket.on('disconnect', () => {
        const indicator = document.getElementById('status-indicator');
        indicator.classList.remove('online');
        indicator.textContent = 'Disconnected';
        document.getElementById('start-button').disabled = true;
    });

    socket.on('metrics_update', (data) => {
        updateMetrics(data);
        updateTrainingProgress(data);
    });

    socket.on('attack_distribution', (data) => {
        updateAttackDistribution(data);
    });

    socket.on('detection', (data) => {
        addDetection(data);
    });

    socket.on('clients_update', (data) => {
        updateClientStatus(data);
    });

    // Start button handler
    document.getElementById('start-button').addEventListener('click', () => {
        socket.emit('start_training');
        document.getElementById('start-button').disabled = true;
    });

    // Initialize charts
    initializeCharts();
});
