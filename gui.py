"""Hockey Player Analysis GUI Application"""
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import logging
import os
import threading
import queue
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import our modules
from src.video import VideoProcessor
from src import DEFAULT_CONFIG, logger

class HockeyAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hockey Player Analysis")
        self.root.geometry("1000x800")
        
        # Initialize variables
        self.video_path = tk.StringVar()
        self.processing = False
        self.current_progress = tk.DoubleVar(value=0)
        
        # Setup GUI components
        self.setup_gui()
        self.setup_text_tags()
        
    def setup_gui(self):
        """Setup main GUI components"""
        # File selection
        file_frame = ttk.LabelFrame(self.root, text="Video Selection", padding="5")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.file_entry = ttk.Entry(file_frame, textvariable=self.video_path)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_btn.pack(side=tk.RIGHT)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding="5")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.current_progress,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack()
        
        # Output section with enhanced text display
        output_frame = ttk.LabelFrame(self.root, text="Analysis Results", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure output text with monospace font and proper width
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            height=20,
            width=80,
            font=('Consolas', 10),
            wrap=tk.NONE,
            background='white',
            foreground='black'
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_btn = ttk.Button(
            button_frame,
            text="Start Processing",
            command=self.start_processing
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear Output",
            command=self.clear_output
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

    def setup_text_tags(self):
        """Setup enhanced text tags for better formatting"""
        # Header style with bold font and center alignment
        self.output_text.tag_configure('header', 
            font=('Consolas', 12, 'bold'),
            justify='center',
            spacing1=10,
            spacing3=10
        )
        
        # Time style with bold and extra spacing
        self.output_text.tag_configure('time', 
            font=('Consolas', 11, 'bold'),
            spacing1=5,
            spacing3=5
        )
        
        # Separator style
        self.output_text.tag_configure('separator', 
            font=('Consolas', 10),
            spacing3=3
        )
        
        # Team styles with appropriate spacing
        self.output_text.tag_configure('black_team', 
            font=('Consolas', 10),
            spacing1=3,
            lmargin1=30,
            lmargin2=30
        )
        
        self.output_text.tag_configure('white_team', 
            font=('Consolas', 10),
            spacing1=3,
            spacing3=10,
            lmargin1=30,
            lmargin2=30
        )
        
    def browse_file(self):
        """Open file dialog to select video"""
        filename = filedialog.askopenfilename(
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_path.set(filename)
            
    def update_progress(self, value):
        """Update progress bar and status"""
        self.current_progress.set(value)
        self.status_label.config(text=f"Processing: {value:.1f}%")
        self.root.update_idletasks()
        
    def add_output(self, message):
        """Add formatted message to output text area with enhanced visuals"""
        lines = message.splitlines()
        for line in lines:
            if "Starting video analysis" in line:
                # Enhanced header with double-line box
                self.output_text.insert(tk.END, "╔══════════════════════════════════════════════════════════╗\n", 'header')
                self.output_text.insert(tk.END, "║              Starting video analysis...                  ║\n", 'header')
                self.output_text.insert(tk.END, "╚══════════════════════════════════════════════════════════╝\n\n", 'header')
            elif line.startswith("Time"):
                # Enhanced timestamp with underline
                self.output_text.insert(tk.END, f"{line}:\n", 'time')
                self.output_text.insert(tk.END, "▔" * 60 + "\n", 'separator')
            elif "Black jersey:" in line:
                # Format team information with proper indentation and alignment
                parts = line.split(":")
                if len(parts) == 2:
                    label = parts[0].strip()
                    numbers = parts[1].strip()
                    self.output_text.insert(tk.END, f"{label}:    {numbers}\n", 'black_team')
            elif "White jersey:" in line:
                # Format team information with proper indentation and alignment
                parts = line.split(":")
                if len(parts) == 2:
                    label = parts[0].strip()
                    numbers = parts[1].strip()
                    self.output_text.insert(tk.END, f"{label}:    {numbers}\n", 'white_team')

        self.output_text.see(tk.END)
        self.root.update_idletasks()
        
    def start_processing(self):
        """Start video processing"""
        if not self.video_path.get():
            self.add_output("Error: No video file selected")
            return
            
        if self.processing:
            self.add_output("Warning: Processing already in progress")
            return
            
        # Disable controls
        self.start_btn.state(['disabled'])
        self.file_entry.state(['disabled'])
        self.browse_btn.state(['disabled'])
        
        # Clear previous output
        self.clear_output()
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(self.video_path.get()), 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Start processing thread
        self.processing = True
        threading.Thread(
            target=self.process_video,
            args=(output_dir,),
            daemon=True
        ).start()
        
    def process_video(self, output_dir):
        """Process video in background thread"""
        try:
            # Initialize video processor
            processor = VideoProcessor(self.video_path.get())
            
            # Redirect stdout to capture print statements
            import io
            stdout = sys.stdout
            output_buffer = io.StringIO()
            sys.stdout = output_buffer
            
            # Process video
            processor.process_video(output_dir, self.update_progress)
            
            # Get captured output
            sys.stdout = stdout
            output = output_buffer.getvalue()
            
            # Display output in GUI
            self.root.after(0, lambda: self.add_output(output))
            
            # Update status on completion
            self.root.after(0, self.processing_complete)
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            self.root.after(0, self.processing_failed)
        finally:
            sys.stdout = stdout
            
    def processing_complete(self):
        """Handle processing completion"""
        self.processing = False
        self.status_label.config(text="Processing Complete")
        self.enable_controls()
        
    def processing_failed(self):
        """Handle processing failure"""
        self.processing = False
        self.status_label.config(text="Processing Failed")
        self.enable_controls()
        
    def enable_controls(self):
        """Re-enable GUI controls"""
        self.start_btn.state(['!disabled'])
        self.file_entry.state(['!disabled'])
        self.browse_btn.state(['!disabled'])
        
    def clear_output(self):
        """Clear output text area"""
        self.output_text.delete(1.0, tk.END)

def main():
    root = tk.Tk()
    app = HockeyAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()