import cv2
import json

class ROIDrawer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.rois = {}  # Store selected ROIs
        self.current_roi = None  # Store current ROI being drawn
        self.drawing = False  # Flag for whether user is drawing ROI
        self.frame = None  # Store current frame
        self.clone = None  # Clone for resetting
        self.current_frame = 0  # Track the current frame index
        self.total_frames = 0  # Store total frame count
        self.cap = None  # VideoCapture object
        self.roi_stage = 'elimination'  # Stage of ROI drawing

    def draw_roi(self, event, x, y, flags, param):
        """ Mouse callback to handle ROI selection """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing ROI
            self.drawing = True
            self.current_roi = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE:
            # Update the ROI being drawn
            if self.drawing:
                self.frame = self.clone.copy()
                cv2.rectangle(self.frame, self.current_roi[0], (x, y), (0, 255, 0), 2)
                cv2.imshow("Frame", self.frame)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finalize ROI
            self.drawing = False
            self.current_roi.append((x, y))
            cv2.rectangle(self.frame, self.current_roi[0], self.current_roi[1], (0, 255, 0), 2)
            cv2.imshow("Frame", self.frame)
            x1, y1 = self.current_roi[0]
            x2, y2 = self.current_roi[1]
            # Store the ROI with current stage name
            self.rois[self.roi_stage] = {
                "top": min(y1, y2),
                "bottom": max(y1, y2),
                "left": min(x1, x2),
                "right": max(x1, x2)
            }
            print(f"ROI for '{self.roi_stage}' saved: {self.rois[self.roi_stage]}")

            # Automatically move to the next ROI stage
            if self.roi_stage == 'elimination':
                self.roi_stage = 'storm'
                print("Select ROI for storm.")
            elif self.roi_stage == 'storm':
                self.roi_stage = 'players'
                print("Select ROI for players.")
            elif self.roi_stage == 'players':
                # Save ROIs automatically once all are selected
                print("All ROIs selected. Saving to 'rois.json'.")
                self.save_rois()
                cv2.destroyAllWindows()
                self.cap.release()

    def update_frame(self, position):
        """Update the displayed frame based on the trackbar position"""
        self.current_frame = position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, self.frame = self.cap.read()
        if ret:
            self.clone = self.frame.copy()
            cv2.imshow("Frame", self.frame)

    def save_rois(self):
        """Save the ROIs to a JSON file"""
        with open("rois.json", "w") as f:
            json.dump(self.rois, f, indent=4)
        print("ROIs saved to 'rois.json'.")

    def select_rois(self):
        # Load video
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set the window in normal mode to avoid zooming issues
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame", 1280, 720)  # Resize window to a reasonable size

        # Set the trackbar to control video frame navigation
        cv2.createTrackbar("Frame", "Frame", 0, self.total_frames - 1, self.update_frame)

        # Initialize frame and mouse callback
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, self.frame = self.cap.read()

        if not ret:
            print("Failed to load video.")
            return

        self.clone = self.frame.copy()  # Clone the frame for resetting during drawing
        cv2.setMouseCallback("Frame", self.draw_roi)

        print("Select ROI for elimination.")
        print("Use the scroll bar to navigate frames.")

        while True:
            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Exiting without saving.")
                break

        cv2.destroyAllWindows()
        self.cap.release()

if __name__ == "__main__":
    video_path = "realvideo.mp4"  # Path to your video
    roi_drawer = ROIDrawer(video_path)
    roi_drawer.select_rois()
