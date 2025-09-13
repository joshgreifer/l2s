export interface DomRefs {
    startGazeDetectionButton: HTMLButtonElement;
    statusDiv: HTMLDivElement;
    indicatorApiAvailableEl: HTMLDivElement;
}

export function getDomRefs(): DomRefs {
    return {
        startGazeDetectionButton: document.querySelector('#startGazeDetectionButton') as HTMLButtonElement,
        statusDiv: document.querySelector('#statusDiv') as HTMLDivElement,
        indicatorApiAvailableEl: document.querySelector('.api-avail-indicator') as HTMLDivElement,
    };
}
