export class UI {
  startGazeDetectionButton: HTMLButtonElement;
  apiIndicator: HTMLDivElement;
  plotsDiv: HTMLDivElement;
  vidCapOverlay: HTMLDivElement;
  overlayCanvas: HTMLCanvasElement;
  vidCap: HTMLVideoElement;
  vidCapContainer: HTMLDivElement;
  landmarkSelector: HTMLDivElement;
  notificationDiv: HTMLDivElement;
  pageTabs: HTMLElement;

  constructor() {
    this.startGazeDetectionButton = document.querySelector('#startGazeDetectionButton') as HTMLButtonElement;
    this.apiIndicator = document.querySelector('.api-avail-indicator') as HTMLDivElement;
    this.plotsDiv = document.querySelector('#plots') as HTMLDivElement;
    this.vidCapOverlay = document.getElementById('vidCapOverlay') as HTMLDivElement;
    this.overlayCanvas = document.querySelector('#overlayCanvas') as HTMLCanvasElement;
    this.vidCap = document.querySelector('#vidCap') as HTMLVideoElement;
    this.vidCapContainer = document.querySelector('.vidcap') as HTMLDivElement;
    this.landmarkSelector = document.querySelector('.landmark_selector') as HTMLDivElement;
    this.notificationDiv = document.querySelector('.notification') as HTMLDivElement;
    this.pageTabs = document.querySelector('.page-tabs') as HTMLElement;
  }
}

export const ui = new UI();
