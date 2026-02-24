"""
Admin Panel for Voice Monitoring
View and manage all active voice monitoring sessions
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from typing import Dict, List
import threading
import time

from voice_monitoring import VoiceMonitor


class AdminPanel:
    """Admin panel to monitor all active voice monitoring sessions."""
    
    # Class variable to store all active monitors
    all_monitors: Dict[str, VoiceMonitor] = {}
    
    def __init__(self):
        """Initialize admin panel window."""
        self.window = tk.Tk()
        self.window.title("Voice Monitoring - Admin Panel")
        self.window.geometry("1000x700")
        self.window.configure(bg="#0a1628")
        
        self.create_widgets()
        self.update_display()
        
    def create_widgets(self):
        """Create admin panel interface."""
        # Header
        header_frame = tk.Frame(self.window, bg="#1a2d4d", height=100)
        header_frame.pack(fill="x", padx=20, pady=20)
        header_frame.pack_propagate(False)
        
        title = tk.Label(
            header_frame,
            text="🎤 Voice Monitoring - Admin Panel",
            font=("Segoe UI", 20, "bold"),
            bg="#1a2d4d",
            fg="#00d4ff",
        )
        title.pack(pady=15)
        
        subtitle = tk.Label(
            header_frame,
            text="Monitor all active voice authentication sessions",
            font=("Segoe UI", 11),
            bg="#1a2d4d",
            fg="#a0b0c0",
        )
        subtitle.pack()
        
        # Statistics Frame
        stats_frame = tk.Frame(self.window, bg="#1a2d4d", relief=tk.SOLID, bd=1)
        stats_frame.pack(fill="x", padx=20, pady=10)
        
        stats_inner = tk.Frame(stats_frame, bg="#1a2d4d")
        stats_inner.pack(pady=15)
        
        self.total_sessions_var = tk.StringVar(value="0")
        self.active_checks_var = tk.StringVar(value="0")
        self.avg_success_var = tk.StringVar(value="0%")
        self.alerts_var = tk.StringVar(value="0")
        self.multi_voice_var = tk.StringVar(value="0")
        
        stats = [
            ("👥 Total Sessions:", self.total_sessions_var),
            ("✅ Total Checks:", self.active_checks_var),
            ("📊 Avg Success:", self.avg_success_var),
            ("🚨 Alerts:", self.alerts_var),
            ("👥👥 Multi-Voice:", self.multi_voice_var),
        ]
        
        for i, (label, var) in enumerate(stats):
            container = tk.Frame(stats_inner, bg="#0d1929", relief=tk.SOLID, bd=1)
            container.grid(row=0, column=i, padx=10)
            
            label_widget = tk.Label(
                container,
                text=label,
                font=("Segoe UI", 10),
                bg="#0d1929",
                fg="#a0b0c0",
                padx=15,
                pady=5,
            )
            label_widget.pack()
            
            value_widget = tk.Label(
                container,
                textvariable=var,
                font=("Segoe UI", 16, "bold"),
                bg="#0d1929",
                fg="#00ff88",
                padx=15,
                pady=5,
            )
            value_widget.pack()
        
        # Sessions Table
        table_frame = tk.LabelFrame(
            self.window,
            text="Active Monitoring Sessions",
            font=("Segoe UI", 12, "bold"),
            bg="#0a1628",
            fg="#00d4ff",
            relief=tk.SOLID,
            bd=1,
        )
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Scrollbar
        scroll_y = tk.Scrollbar(table_frame)
        scroll_y.pack(side="right", fill="y")
        
        # Treeview
        columns = ("User ID", "Status", "Checks", "Success", "Failed", "Rate", "Last Check")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            yscrollcommand=scroll_y.set,
            height=15,
        )
        
        # Configure columns
        self.tree.column("User ID", width=150)
        self.tree.column("Status", width=100)
        self.tree.column("Checks", width=80)
        self.tree.column("Success", width=80)
        self.tree.column("Failed", width=80)
        self.tree.column("Rate", width=100)
        self.tree.column("Last Check", width=150)
        
        for col in columns:
            self.tree.heading(col, text=col)
        
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        scroll_y.config(command=self.tree.yview)
        
        # Configure treeview style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#0d1929",
            foreground="white",
            fieldbackground="#0d1929",
            borderwidth=0,
        )
        style.configure("Treeview.Heading", background="#1a2d4d", foreground="#00d4ff", font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", "#00d4ff")])
        
        # Context menu for right-click
        self.tree.bind("<Button-3>", self.show_context_menu)
        self.tree.bind("<Double-1>", self.show_user_details)
        
        # Activity Log
        log_frame = tk.LabelFrame(
            self.window,
            text="Recent Activity",
            font=("Segoe UI", 11, "bold"),
            bg="#0a1628",
            fg="#00d4ff",
            relief=tk.SOLID,
            bd=1,
        )
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.log_text = tk.Text(
            log_frame,
            height=8,
            font=("Consolas", 9),
            bg="#0d1929",
            fg="#ffffff",
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Control Buttons
        button_frame = tk.Frame(self.window, bg="#0a1628")
        button_frame.pack(fill="x", padx=20, pady=15)
        
        refresh_btn = tk.Button(
            button_frame,
            text="🔄 Refresh",
            font=("Segoe UI", 11, "bold"),
            bg="#0088cc",
            fg="white",
            activebackground="#00aaff",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.manual_refresh,
            padx=20,
            pady=8,
        )
        refresh_btn.pack(side="left", padx=5)
        
        clear_log_btn = tk.Button(
            button_frame,
            text="🗑️ Clear Log",
            font=("Segoe UI", 11, "bold"),
            bg="#666666",
            fg="white",
            activebackground="#888888",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.clear_log,
            padx=20,
            pady=8,
        )
        clear_log_btn.pack(side="left", padx=5)
        
        close_btn = tk.Button(
            button_frame,
            text="✖️ Close",
            font=("Segoe UI", 11, "bold"),
            bg="#ff4444",
            fg="white",
            activebackground="#ff6666",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.window.quit,
            padx=20,
            pady=8,
        )
        close_btn.pack(side="right", padx=5)
        
        self.log("Admin panel initialized")
    
    def update_display(self):
        """Update display with latest monitoring data."""
        try:
            # Update statistics
            total_sessions = len(AdminPanel.all_monitors)
            total_checks = 0
            total_success = 0
            total_alerts = 0
            total_multi_voice = 0
            
            for monitor in AdminPanel.all_monitors.values():
                status = monitor.get_monitoring_status()
                total_checks += status['total_checks']
                total_success += status['successful_checks']
                if status['consecutive_failures'] >= 3:
                    total_alerts += 1
                # Count multiple speaker detections
                history = monitor.get_verification_history()
                for check in history:
                    if check.get('multiple_speakers', False):
                        total_multi_voice += 1
            
            avg_success = (total_success / total_checks * 100) if total_checks > 0 else 0
            
            self.total_sessions_var.set(str(total_sessions))
            self.active_checks_var.set(str(total_checks))
            self.avg_success_var.set(f"{avg_success:.1f}%")
            self.alerts_var.set(str(total_alerts))
            self.multi_voice_var.set(str(total_multi_voice))
            
            # Update sessions table
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Add current sessions
            for user_id, monitor in AdminPanel.all_monitors.items():
                status = monitor.get_monitoring_status()
                
                monitoring_status = "🟢 Active" if status['is_monitoring'] else "⚪ Stopped"
                success_rate = f"{status['success_rate']:.1%}"
                
                last_check = "Never"
                if status['last_check']:
                    last_time = datetime.fromisoformat(status['last_check']['timestamp'])
                    last_check = last_time.strftime("%H:%M:%S")
                
                # Color code based on success rate
                if status['success_rate'] >= 0.9:
                    tag = "good"
                elif status['success_rate'] >= 0.7:
                    tag = "warning"
                else:
                    tag = "danger"
                
                self.tree.insert(
                    "",
                    "end",
                    values=(
                        user_id,
                        monitoring_status,
                        status['total_checks'],
                        status['successful_checks'],
                        status['failed_checks'],
                        success_rate,
                        last_check,
                    ),
                    tags=(tag,)
                )
            
            # Configure row colors
            self.tree.tag_configure("good", foreground="#00ff88")
            self.tree.tag_configure("warning", foreground="#ffaa00")
            self.tree.tag_configure("danger", foreground="#ff4444")
            
        except Exception as e:
            self.log(f"Error updating display: {str(e)}")
        
        # Schedule next update
        self.window.after(2000, self.update_display)  # Update every 2 seconds
    
    def show_context_menu(self, event):
        """Show context menu on right-click."""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            menu = tk.Menu(self.window, tearoff=0, bg="#1a2d4d", fg="white")
            menu.add_command(label="View Details", command=self.show_user_details)
            menu.add_command(label="Stop Monitoring", command=self.stop_selected_user)
            menu.add_separator()
            menu.add_command(label="View History", command=self.show_history)
            menu.post(event.x_root, event.y_root)
    
    def show_user_details(self, event=None):
        """Show detailed information for selected user."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        user_id = item['values'][0]
        
        if user_id not in AdminPanel.all_monitors:
            return
        
        monitor = AdminPanel.all_monitors[user_id]
        status = monitor.get_monitoring_status()
        
        details = (
            f"User: {user_id}\n\n"
            f"Total Checks: {status['total_checks']}\n"
            f"Successful: {status['successful_checks']}\n"
            f"Failed: {status['failed_checks']}\n"
            f"Success Rate: {status['success_rate']:.1%}\n"
            f"Consecutive Failures: {status['consecutive_failures']}\n"
            f"Status: {'Monitoring' if status['is_monitoring'] else 'Stopped'}\n"
        )
        
        messagebox.showinfo(f"Details - {user_id}", details, parent=self.window)
        self.log(f"Viewed details for {user_id}")
    
    def stop_selected_user(self):
        """Stop monitoring for selected user."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        user_id = item['values'][0]
        
        if user_id in AdminPanel.all_monitors:
            monitor = AdminPanel.all_monitors[user_id]
            monitor.stop_monitoring()
            del AdminPanel.all_monitors[user_id]
            self.log(f"Stopped monitoring for {user_id}")
            messagebox.showinfo("Stopped", f"Monitoring stopped for {user_id}", parent=self.window)
    
    def show_history(self):
        """Show full history for selected user."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        user_id = item['values'][0]
        
        if user_id not in AdminPanel.all_monitors:
            return
        
        monitor = AdminPanel.all_monitors[user_id]
        history = monitor.get_verification_history()
        
        # Create history window
        history_window = tk.Toplevel(self.window)
        history_window.title(f"History - {user_id}")
        history_window.geometry("700x500")
        history_window.configure(bg="#0a1628")
        
        text = tk.Text(history_window, font=("Consolas", 9), bg="#0d1929", fg="white")
        text.pack(fill="both", expand=True, padx=10, pady=10)
        
        for entry in history:
            timestamp = entry.get('timestamp', 'Unknown')
            verified = "✅ PASS" if entry.get('verified') else "❌ FAIL"
            similarity = entry.get('similarity', 0)
            text.insert("end", f"[{timestamp}] {verified} - Similarity: {similarity:.2%}\n")
        
        text.config(state="disabled")
        self.log(f"Viewed history for {user_id} ({len(history)} entries)")
    
    def manual_refresh(self):
        """Manually refresh display."""
        self.log("Manual refresh triggered")
        # Update will happen automatically from scheduled update
    
    def clear_log(self):
        """Clear activity log."""
        self.log_text.delete("1.0", "end")
    
    def log(self, message: str):
        """Add message to activity log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert("end", log_entry)
        self.log_text.see("end")
    
    def run(self):
        """Run the admin panel."""
        self.window.mainloop()
    
    @classmethod
    def add_monitor(cls, user_id: str, monitor: VoiceMonitor):
        """Add a monitor to the admin panel tracking."""
        cls.all_monitors[user_id] = monitor
    
    @classmethod
    def remove_monitor(cls, user_id: str):
        """Remove a monitor from tracking."""
        if user_id in cls.all_monitors:
            del cls.all_monitors[user_id]


if __name__ == "__main__":
    print("Starting Admin Panel...")
    print("Note: Start monitoring sessions from the main app first")
    print("Then open this admin panel to view them")
    
    panel = AdminPanel()
    panel.run()
