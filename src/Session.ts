/*
A session consists of a Hypnotherapist and Subject interacting.  The therapist speaks,  and the subject responds either
 by speaking, or by performing non-verbal actions (e.g. breathing deeply, closing eyes).

 This app continuously records the subject's state using the camera and microphone:
 Gaze point (which part of the screen they're looking at)
 Eye state (open, closed, blink)
 Heart rate
 Breathing rate
 Facial expression
 Verbal responses (what the subject says)
 */
import {Subject} from "./Subject";


export class Session {
    notificationDiv: HTMLDivElement

    constructor(private subject: Subject) {
        this.notificationDiv = <HTMLDivElement>document.querySelector('.notification');
    }

    private Notify(message: string): void {
        this.notificationDiv.innerText = message;
        this.notificationDiv.style.opacity = "1";

        window.setTimeout( ()=> {
            this.notificationDiv.style.opacity = "0";
        }, 3000);
    }
    public async Run() {

        const handleUtteranceKeyboardShortcuts = async (evt: KeyboardEvent) => {
            if (evt.key === 's') {
                this.Notify("Saving model.");
                const success = await s.SaveGazeDetectorCalibration();
                this.Notify(success ? "Saved model." : "Failed to save model.");

            } else if (evt.key === 'c') {
                if (s.isGazeCalibrationActive)
                    await s.StopGazeDetectorCalibration();
                else
                    await s.StartGazeDetectorCalibration();

                this.Notify(s.isGazeCalibrationActive ? "Calibration started." : "Calibration stopped.");

            }

        };
        window.addEventListener('keyup',handleUtteranceKeyboardShortcuts);
        const s = this.subject;

        await s.StartGazeDetection();
    }



}