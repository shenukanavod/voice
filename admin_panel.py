"""
Admin Panel for Voice Monitoring
View and manage all active voice monitoring sessions
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import threading
import time

from voice_monitoring import VoiceMonitor
from db import VoiceDatabase
from app.config import settings

# Helper function to convert UTC to local time
def utc_to_local(utc_dt):
    """Convert UTC datetime to local timezone."""
    if utc_dt is None:
        return datetime.now()
    
    # If it's a naive datetime (no timezone info), assume it's UTC
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    
    # Convert to local time
    local_dt = utc_dt.astimezone()
    return local_dt

# Initialize database connection
try:
    db_instance = VoiceDatabase(settings.MONGODB_URL)
    print("✅ Admin panel connected to database")
except Exception as e:
    print(f"⚠️  Admin panel database connection failed: {e}")
    db_instance = None


class AdminPanel:
    """Admin panel to monitor all active voice monitoring sessions."""
    
    # Class variable to store all active monitors
    all_monitors: Dict[str, VoiceMonitor] = {}
    
    def __init__(self):
        """Initialize admin panel window."""
        self.window = tk.Tk()
        self.window.title("Voice Monitoring - Admin Panel")
        self.window.geometry("1200x800")
        self.window.configure(bg="#0a1628")
        
        self.create_widgets()
        self.update_display()
        self.load_database_logs()  # Load database logs on startup
        self.load_login_history()  # Load login history on startup
        
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
            text="Monitor all active voice authentication sessions and view database history",
            font=("Segoe UI", 11),
            bg="#1a2d4d",
            fg="#a0b0c0",
        )
        subtitle.pack()
        
        # Create notebook (tabbed interface)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#0a1628", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1a2d4d", foreground="#ffffff", padding=[20, 10])
        style.map("TNotebook.Tab", background=[("selected", "#00d4ff")], foreground=[("selected", "#000000")])
        
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Tab 1: Live Monitoring
        self.live_tab = tk.Frame(self.notebook, bg="#0a1628")
        self.notebook.add(self.live_tab, text="🔴 Live Monitoring")
        
        # Tab 2: Database History
        self.db_tab = tk.Frame(self.notebook, bg="#0a1628")
        self.notebook.add(self.db_tab, text="📊 Database History")
        
        # Tab 3: Login History
        self.login_tab = tk.Frame(self.notebook, bg="#0a1628")
        self.notebook.add(self.login_tab, text="🔐 Login History")
        
        # Create widgets for each tab
        self.create_live_monitoring_tab()
        self.create_database_history_tab()
        self.create_login_history_tab()
        
        # Bind tab change event to refresh data
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Control Buttons (at bottom)
        button_frame = tk.Frame(self.window, bg="#0a1628")
        button_frame.pack(fill="x", padx=20, pady=15)
        
        refresh_btn = tk.Button(
            button_frame,
            text="🔄 Refresh All",
            font=("Segoe UI", 11, "bold"),
            bg="#0088cc",
            fg="white",
            activebackground="#00aaff",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.refresh_all_data,
            padx=20,
            pady=8,
        )
        refresh_btn.pack(side="left", padx=5)
        
        # Status label for loading feedback
        self.status_label = tk.Label(
            button_frame,
            text="",
            font=("Segoe UI", 10),
            bg="#0a1628",
            fg="#00ff88",
        )
        self.status_label.pack(side="left", padx=20)
        
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
    
    def create_live_monitoring_tab(self):
        """Create the live monitoring tab."""
        # Statistics Frame
        stats_frame = tk.Frame(self.live_tab, bg="#1a2d4d", relief=tk.SOLID, bd=1)
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
            self.live_tab,
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
            height=10,
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
            self.live_tab,
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
            height=5,
            font=("Consolas", 9),
            bg="#0d1929",
            fg="#ffffff",
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    def create_database_history_tab(self):
        """Create the database history tab."""
        # Time filter frame
        filter_frame = tk.Frame(self.db_tab, bg="#1a2d4d", relief=tk.SOLID, bd=1)
        filter_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(
            filter_frame,
            text="⏱️ Time Range:",
            font=("Segoe UI", 11, "bold"),
            bg="#1a2d4d",
            fg="#00d4ff",
        ).pack(side="left", padx=10, pady=10)
        
        # Store current selection and button references
        self.selected_time_filter = "All Time"
        self.time_filter_buttons = {}
        
        time_options = ["15 Minutes", "1 Hour", "6 Hours", "24 Hours", "All Time"]
        
        def create_time_button(option):
            """Create a time filter button."""
            def on_click():
                # Update selected filter
                self.selected_time_filter = option
                print(f"\n🔄 TIME FILTER: {option}")
                
                # Update button colors
                for opt, btn in self.time_filter_buttons.items():
                    if opt == option:
                        btn.config(bg="#00d4ff", fg="#000000", relief=tk.SUNKEN)
                    else:
                        btn.config(bg="#0d1929", fg="#ffffff", relief=tk.RAISED)
                
                # Load data with new filter
                self.load_database_logs()
            
            btn = tk.Button(
                filter_frame,
                text=option,
                font=("Segoe UI", 10, "bold"),
                bg="#00d4ff" if option == "All Time" else "#0d1929",
                fg="#000000" if option == "All Time" else "#ffffff",
                activebackground="#00aaff",
                activeforeground="#000000",
                relief=tk.SUNKEN if option == "All Time" else tk.RAISED,
                cursor="hand2",
                command=on_click,
                padx=15,
                pady=5,
            )
            btn.pack(side="left", padx=3)
            self.time_filter_buttons[option] = btn
            return btn
        
        for option in time_options:
            create_time_button(option)
        
        # Database Statistics
        db_stats_frame = tk.Frame(self.db_tab, bg="#1a2d4d", relief=tk.SOLID, bd=1)
        db_stats_frame.pack(fill="x", padx=20, pady=10)
        
        stats_inner = tk.Frame(db_stats_frame, bg="#1a2d4d")
        stats_inner.pack(pady=15)
        
        self.db_total_logs_var = tk.StringVar(value="0")
        self.db_voice_checks_var = tk.StringVar(value="0")
        self.db_alerts_var = tk.StringVar(value="0")
        self.db_sessions_var = tk.StringVar(value="0")
        
        stats = [
            ("📝 Total Logs:", self.db_total_logs_var),
            ("🔍 Voice Checks:", self.db_voice_checks_var),
            ("🚨 Alerts:", self.db_alerts_var),
            ("👥 Sessions:", self.db_sessions_var),
        ]
        
        for i, (label, var) in enumerate(stats):
            container = tk.Frame(stats_inner, bg="#0d1929", relief=tk.SOLID, bd=1)
            container.grid(row=0, column=i, padx=15)
            
            tk.Label(
                container,
                text=label,
                font=("Segoe UI", 10),
                bg="#0d1929",
                fg="#a0b0c0",
                padx=20,
                pady=5,
            ).pack()
            
            tk.Label(
                container,
                textvariable=var,
                font=("Segoe UI", 16, "bold"),
                bg="#0d1929",
                fg="#00ff88",
                padx=20,
                pady=5,
            ).pack()
        
        # Monitoring Data Notebook (sub-tabs)
        data_notebook = ttk.Notebook(self.db_tab)
        data_notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Voice Checks Tab
        checks_tab = tk.Frame(data_notebook, bg="#0a1628")
        data_notebook.add(checks_tab, text="🔍 Voice Checks")
        
        checks_scroll = tk.Scrollbar(checks_tab)
        checks_scroll.pack(side="right", fill="y")
        
        checks_columns = ("Time", "User", "Result", "Similarity", "Threshold", "Speaker Status")
        self.checks_tree = ttk.Treeview(
            checks_tab,
            columns=checks_columns,
            show="headings",
            yscrollcommand=checks_scroll.set,
        )
        
        self.checks_tree.column("Time", width=150)
        self.checks_tree.column("User", width=150)
        self.checks_tree.column("Result", width=200)
        self.checks_tree.column("Similarity", width=100)
        self.checks_tree.column("Threshold", width=100)
        self.checks_tree.column("Speaker Status", width=180)
        
        for col in checks_columns:
            self.checks_tree.heading(col, text=col)
        
        self.checks_tree.pack(fill="both", expand=True, padx=10, pady=10)
        checks_scroll.config(command=self.checks_tree.yview)
        
        # Security Alerts Tab
        alerts_tab = tk.Frame(data_notebook, bg="#0a1628")
        data_notebook.add(alerts_tab, text="🚨 Security Alerts")
        
        alerts_scroll = tk.Scrollbar(alerts_tab)
        alerts_scroll.pack(side="right", fill="y")
        
        alerts_columns = ("Time", "User", "Alert Type", "Details")
        self.alerts_tree = ttk.Treeview(
            alerts_tab,
            columns=alerts_columns,
            show="headings",
            yscrollcommand=alerts_scroll.set,
        )
        
        self.alerts_tree.column("Time", width=150)
        self.alerts_tree.column("User", width=150)
        self.alerts_tree.column("Alert Type", width=200)
        self.alerts_tree.column("Details", width=400)
        
        for col in alerts_columns:
            self.alerts_tree.heading(col, text=col)
        
        self.alerts_tree.pack(fill="both", expand=True, padx=10, pady=10)
        alerts_scroll.config(command=self.alerts_tree.yview)
        
        # Sessions Tab
        sessions_tab = tk.Frame(data_notebook, bg="#0a1628")
        data_notebook.add(sessions_tab, text="👥 Sessions")
        
        sessions_scroll = tk.Scrollbar(sessions_tab)
        sessions_scroll.pack(side="right", fill="y")
        
        sessions_columns = ("Time", "User", "Event", "Total Checks", "Success Rate")
        self.sessions_tree = ttk.Treeview(
            sessions_tab,
            columns=sessions_columns,
            show="headings",
            yscrollcommand=sessions_scroll.set,
        )
        
        self.sessions_tree.column("Time", width=150)
        self.sessions_tree.column("User", width=150)
        self.sessions_tree.column("Event", width=150)
        self.sessions_tree.column("Total Checks", width=120)
        self.sessions_tree.column("Success Rate", width=120)
        
        for col in sessions_columns:
            self.sessions_tree.heading(col, text=col)
        
        self.sessions_tree.pack(fill="both", expand=True, padx=10, pady=10)
        sessions_scroll.config(command=self.sessions_tree.yview)
    
    def load_database_logs(self):
        """Load monitoring logs from database."""
        # Show loading status
        if hasattr(self, 'status_label'):
            self.status_label.config(text="⏳ Loading database logs...", fg="#ffaa00")
            self.window.update_idletasks()
        
        if not db_instance:
            self.log("❌ Database not connected", level="ERROR")
            if hasattr(self, 'status_label'):
                self.status_label.config(text="❌ Database not connected", fg="#ff4444")
            messagebox.showwarning(
                "Database Connection",
                "Database is not connected. Cannot load monitoring history.\n\nPlease check your MongoDB connection.",
                parent=self.window
            )
            return
        
        try:
            # Get selected time filter
            time_filter = self.selected_time_filter if hasattr(self, 'selected_time_filter') else "All Time"
            print(f"\n🔍 Loading database logs with time filter: {time_filter}")
            self.log(f"Loading logs for: {time_filter}")
            hours = None
            
            if time_filter == "15 Minutes":
                hours = 0.25
            elif time_filter == "1 Hour":
                hours = 1
            elif time_filter == "6 Hours":
                hours = 6
            elif time_filter == "24 Hours":
                hours = 24
            
            # Build custom filter for time range (use utcnow for comparison)
            custom_filter = None
            if hours is not None:
                # Use naive datetime matching MongoDB's utcnow() storage
                since = datetime.utcnow() - timedelta(hours=hours)
                custom_filter = {'timestamp': {'$gte': since}}
                print(f"⏰ Time filter applied: {hours} hours (since {since})")
            else:
                print(f"⏰ No time filter (All Time selected)")
            
            # Debug: Check total logs without filter
            if db_instance:
                total_in_db = db_instance.count_logs()
                print(f"📊 Total logs in database: {total_in_db}")
                
                # Show sample data if available
                if total_in_db > 0:
                    sample_logs = db_instance.get_logs(limit=3)
                    print(f"📊 Sample modules in database:")
                    for log in sample_logs:
                        print(f"   - {log.get('module', 'unknown')}: {log.get('message', '')[:50]}")
                
                if custom_filter:
                    print(f"📊 Custom filter: {custom_filter}")
            
            # Get voice checks
            print(f"📊 Querying voice checks...")
            print(f"   Module: 'voice_monitoring_check'")
            print(f"   Custom filter: {custom_filter}")
            print(f"   Filter type: {type(custom_filter)}")
            if custom_filter and 'timestamp' in custom_filter:
                print(f"   Timestamp filter: {custom_filter['timestamp']}")
            
            voice_checks = db_instance.get_logs(
                module="voice_monitoring_check",
                custom_filter=custom_filter,
                limit=1000
            )
            
            # If no results with filter, try without filter to debug
            if len(voice_checks) == 0 and custom_filter:
                print(f"⚠️ No results with filter. Debugging...")
                all_voice_checks = db_instance.get_logs(
                    module="voice_monitoring_check",
                    limit=5
                )
                print(f"   Found {len(all_voice_checks)} voice checks total (without filter)")
                if len(all_voice_checks) > 0:
                    # Show sample timestamps
                    print(f"   Sample timestamps from database:")
                    for i, sample in enumerate(all_voice_checks[:3]):
                        sample_time = sample.get('timestamp')
                        print(f"   [{i+1}] {sample_time} (type: {type(sample_time).__name__})")
                    print(f"   Filter expects times >= {custom_filter.get('timestamp', {}).get('$gte')}")
                    print(f"   Current UTC time: {datetime.utcnow()}")
            
            print(f"   Found {len(voice_checks)} voice check logs")
            
            # Clear and populate voice checks
            for item in self.checks_tree.get_children():
                self.checks_tree.delete(item)
            
            check_count = 0
            for log in voice_checks:
                timestamp_obj = log.get('timestamp', datetime.utcnow())
                # Handle both datetime objects and strings
                if isinstance(timestamp_obj, str):
                    try:
                        from dateutil import parser
                        timestamp_obj = parser.parse(timestamp_obj)
                    except:
                        timestamp_obj = datetime.utcnow()
                
                # Convert UTC to local time for display
                local_time = utc_to_local(timestamp_obj)
                timestamp = local_time.strftime("%Y-%m-%d %H:%M:%S")
                user = log.get('user_id', 'Unknown')
                extra = log.get('extra_data', {})
                
                verified = extra.get('verified', False)
                speaker_status = extra.get('speaker_status', 'Unknown')
                result = f"✅ {speaker_status}" if verified else f"❌ {speaker_status}"
                similarity = extra.get('similarity', 0)
                threshold = extra.get('threshold', 0)
                
                # Determine tag color based on speaker status
                if verified:
                    tag = "same_speaker"
                elif extra.get('multiple_speakers', False):
                    tag = "multi_speaker"
                else:
                    tag = "different_speaker"
                
                self.checks_tree.insert(
                    "",
                    "end",
                    values=(timestamp, user, result, f"{similarity:.1%}", f"{threshold:.1%}", speaker_status),
                    tags=(tag,)
                )
                check_count += 1
            
            self.checks_tree.tag_configure("same_speaker", foreground="#00ff88")
            self.checks_tree.tag_configure("different_speaker", foreground="#ff4444")
            self.checks_tree.tag_configure("multi_speaker", foreground="#ffaa00")
            
            # Get security alerts
            print(f"🚨 Querying security alerts with filter: {custom_filter}")
            alerts = db_instance.get_logs(
                module="voice_monitoring_alert",
                custom_filter=custom_filter,
                limit=1000
            )
            print(f"   Found {len(alerts)} alert logs")
            
            # Clear and populate alerts
            for item in self.alerts_tree.get_children():
                self.alerts_tree.delete(item)
            
            alert_count = 0
            for log in alerts:
                timestamp_obj = log.get('timestamp', datetime.utcnow())
                if isinstance(timestamp_obj, str):
                    try:
                        from dateutil import parser
                        timestamp_obj = parser.parse(timestamp_obj)
                    except:
                        timestamp_obj = datetime.utcnow()
                
                # Convert UTC to local time for display
                local_time = utc_to_local(timestamp_obj)
                timestamp = local_time.strftime("%Y-%m-%d %H:%M:%S")
                user = log.get('user_id', 'Unknown')
                message = log.get('message', '')
                extra = log.get('extra_data', {})
                
                alert_type = extra.get('alert_type', 'Unknown')
                details = f"Failures: {extra.get('consecutive_failures', 0)}, Similarity: {extra.get('last_similarity', 0):.1%}"
                
                self.alerts_tree.insert(
                    "",
                    "end",
                    values=(timestamp, user, alert_type, details),
                    tags=("alert",)
                )
                alert_count += 1
            
            self.alerts_tree.tag_configure("alert", foreground="#ff4444")
            
            # Get sessions
            print(f"👥 Querying sessions with filter: {custom_filter}")
            sessions = db_instance.get_logs(
                module="admin_panel_session",
                custom_filter=custom_filter,
                limit=1000
            )
            print(f"   Found {len(sessions)} session logs")
            
            # Clear and populate sessions
            for item in self.sessions_tree.get_children():
                self.sessions_tree.delete(item)
            
            session_count = 0
            for log in sessions:
                timestamp_obj = log.get('timestamp', datetime.utcnow())
                if isinstance(timestamp_obj, str):
                    try:
                        from dateutil import parser
                        timestamp_obj = parser.parse(timestamp_obj)
                    except:
                        timestamp_obj = datetime.utcnow()
                
                # Convert UTC to local time for display
                local_time = utc_to_local(timestamp_obj)
                timestamp = local_time.strftime("%Y-%m-%d %H:%M:%S")
                user = log.get('user_id', 'Unknown')
                extra = log.get('extra_data', {})
                
                event = extra.get('action', 'Unknown')
                total_checks = extra.get('total_checks', '-')
                success_rate = extra.get('success_rate', 0)
                success_rate_str = f"{success_rate:.1%}" if isinstance(success_rate, (int, float)) else "-"
                
                self.sessions_tree.insert(
                    "",
                    "end",
                    values=(timestamp, user, event, total_checks, success_rate_str),
                )
                session_count += 1
            
            # Update statistics
            total_logs = check_count + alert_count + session_count
            self.db_total_logs_var.set(str(total_logs))
            self.db_voice_checks_var.set(str(check_count))
            self.db_alerts_var.set(str(alert_count))
            self.db_sessions_var.set(str(session_count))
            
            # Only show monitoring-specific logs
            print(f"\n✅ Loaded {total_logs} monitoring logs ({time_filter})")
            print(f"   - Voice Checks: {check_count}")
            print(f"   - Alerts: {alert_count}")
            print(f"   - Sessions: {session_count}\n")
            self.log(f"✅ Loaded {total_logs} logs for {time_filter}")
            
            # Update status label
            if hasattr(self, 'status_label'):
                self.status_label.config(
                    text=f"✅ Loaded {total_logs} logs ({time_filter})", 
                    fg="#00ff88"
                )
            
            # Show message if no monitoring data
            if total_logs == 0:
                self.log("⚠️ No monitoring activity found. Start voice monitoring to see data.")
                if hasattr(self, 'status_label'):
                    self.status_label.config(text="⚠️ No data found", fg="#ffaa00")
                if time_filter != "All Time":
                    self.log(f"💡 Tip: Try selecting 'All Time' to see all historical data.")
                else:
                    self.log(f"💡 Database is connected but empty. Voice checks will appear here after monitoring starts.")
            
        except Exception as e:
            self.log(f"Error loading database logs: {str(e)}", level="ERROR")
            print(f"Database load error: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"❌ Error: {str(e)[:50]}", fg="#ff4444")
    
    def create_login_history_tab(self):
        """Create the login history tab."""
        # Time filter frame
        filter_frame = tk.Frame(self.login_tab, bg="#1a2d4d", relief=tk.SOLID, bd=1)
        filter_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(
            filter_frame,
            text="⏱️ Time Range:",
            font=("Segoe UI", 11, "bold"),
            bg="#1a2d4d",
            fg="#00d4ff",
        ).pack(side="left", padx=10, pady=10)
        
        # Store current selection and button references
        self.selected_login_time_filter = "24 Hours"
        self.login_time_filter_buttons = {}
        
        time_options = ["1 Hour", "6 Hours", "24 Hours", "7 Days", "All Time"]
        
        def create_login_time_button(option):
            """Create a login time filter button."""
            def on_click():
                # Update selected filter
                self.selected_login_time_filter = option
                print(f"\n🔄 LOGIN TIME FILTER: {option}")
                
                # Update button colors
                for opt, btn in self.login_time_filter_buttons.items():
                    if opt == option:
                        btn.config(bg="#00d4ff", fg="#000000", relief=tk.SUNKEN)
                    else:
                        btn.config(bg="#0d1929", fg="#ffffff", relief=tk.RAISED)
                
                # Load data with new filter
                self.load_login_history()
            
            btn = tk.Button(
                filter_frame,
                text=option,
                font=("Segoe UI", 10, "bold"),
                bg="#00d4ff" if option == "24 Hours" else "#0d1929",
                fg="#000000" if option == "24 Hours" else "#ffffff",
                activebackground="#00aaff",
                activeforeground="#000000",
                relief=tk.SUNKEN if option == "24 Hours" else tk.RAISED,
                cursor="hand2",
                command=on_click,
                padx=15,
                pady=5,
            )
            btn.pack(side="left", padx=3)
            self.login_time_filter_buttons[option] = btn
            return btn
        
        for option in time_options:
            create_login_time_button(option)
        
        # Login table
        table_frame = tk.LabelFrame(
            self.login_tab,
            text="Authentication Attempts",
            font=("Segoe UI", 12, "bold"),
            bg="#0a1628",
            fg="#00d4ff",
            relief=tk.SOLID,
            bd=1,
        )
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        scroll_y = tk.Scrollbar(table_frame)
        scroll_y.pack(side="right", fill="y")
        
        login_columns = ("Date", "Time", "User ID", "Status", "Similarity", "Result")
        self.login_tree = ttk.Treeview(
            table_frame,
            columns=login_columns,
            show="headings",
            yscrollcommand=scroll_y.set,
        )
        
        self.login_tree.column("Date", width=120)
        self.login_tree.column("Time", width=100)
        self.login_tree.column("User ID", width=150)
        self.login_tree.column("Status", width=120)
        self.login_tree.column("Similarity", width=100)
        self.login_tree.column("Result", width=300)
        
        for col in login_columns:
            self.login_tree.heading(col, text=col)
        
        self.login_tree.pack(fill="both", expand=True, padx=10, pady=10)
        scroll_y.config(command=self.login_tree.yview)
        
    def load_login_history(self):
        """Load login/authentication history from database."""
        if not db_instance:
            return
        
        try:
            # Get selected time filter
            time_filter = self.selected_login_time_filter if hasattr(self, 'selected_login_time_filter') else "24 Hours"
            print(f"\n🔐 Loading login history with time filter: {time_filter}")
            self.log(f"Loading login history for: {time_filter}")
            hours = None
            
            if time_filter == "1 Hour":
                hours = 1
            elif time_filter == "6 Hours":
                hours = 6
            elif time_filter == "24 Hours":
                hours = 24
            elif time_filter == "7 Days":
                hours = 168
            
            custom_filter = None
            if hours is not None:
                since = datetime.utcnow() - timedelta(hours=hours)
                custom_filter = {'timestamp': {'$gte': since}}
                print(f"⏰ Time filter applied: {hours} hours (since {since})")
            else:
                print(f"⏰ No time filter (All Time selected)")
            
            # Get login attempts
            print(f"🔍 Querying login attempts...")
            login_attempts = db_instance.get_logs(
                module="authentication_login",
                custom_filter=custom_filter,
                limit=1000
            )
            
            # Clear and populate login history
            for item in self.login_tree.get_children():
                self.login_tree.delete(item)
            
            for log in login_attempts:
                timestamp_obj = log.get('timestamp', datetime.utcnow())
                if isinstance(timestamp_obj, str):
                    try:
                        from dateutil import parser
                        timestamp_obj = parser.parse(timestamp_obj)
                    except:
                        timestamp_obj = datetime.utcnow()
                
                # Convert UTC to local time for display
                local_time = utc_to_local(timestamp_obj)
                date_str = local_time.strftime("%Y-%m-%d")
                time_str = local_time.strftime("%H:%M:%S")
                user = log.get('user_id', 'Unknown')
                extra = log.get('extra_data', {})
                
                status = extra.get('status', 'UNKNOWN')
                similarity = extra.get('similarity', 0)
                message = log.get('message', '')
                
                tag = "success" if status == "SUCCESS" else "failed"
                
                self.login_tree.insert(
                    "",
                    "end",
                    values=(date_str, time_str, user, status, f"{similarity:.1%}", message),
                    tags=(tag,)
                )
            
            self.login_tree.tag_configure("success", foreground="#00ff88")
            self.login_tree.tag_configure("failed", foreground="#ff4444")
            
            print(f"✅ Loaded {len(login_attempts)} login attempts ({time_filter})\n")
            self.log(f"✅ Loaded {len(login_attempts)} login attempts for {time_filter}")
        
        except Exception as e:
            self.log(f"Error loading login history: {str(e)}", level="ERROR")
            print(f"Login history load error: {e}")
    
    def refresh_all_data(self):
        """Refresh both live and database data."""
        self.load_database_logs()
        self.load_login_history()
        self.log("All data refreshed")
    
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
            
            # Save statistics to database periodically (every 10 updates = 20 seconds)
            if not hasattr(self, '_update_counter'):
                self._update_counter = 0
            self._update_counter += 1
            
            if self._update_counter % 10 == 0 and db_instance:
                try:
                    db_instance.save_log(
                        level="INFO",
                        message=f"Admin panel statistics update",
                        module="admin_panel_stats",
                        extra_data={
                            "total_sessions": total_sessions,
                            "total_checks": total_checks,
                            "total_success": total_success,
                            "total_alerts": total_alerts,
                            "total_multi_voice": total_multi_voice,
                            "avg_success_rate": avg_success / 100,
                            "active_users": list(AdminPanel.all_monitors.keys())
                        }
                    )
                except Exception as e:
                    print(f"⚠️  Failed to save stats to database: {e}")
            
            # Update sessions table
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Add current sessions and save to database
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
                
                # Save each active session to database history (every 5 updates = 10 seconds)
                if self._update_counter % 5 == 0 and db_instance and status['is_monitoring']:
                    try:
                        db_instance.save_log(
                            level="INFO",
                            message=f"Monitoring session active for {user_id}",
                            module="admin_panel_session",
                            user_id=user_id,
                            extra_data={
                                "action": "Session Update",
                                "total_checks": int(status['total_checks']),
                                "successful_checks": int(status['successful_checks']),
                                "failed_checks": int(status['failed_checks']),
                                "success_rate": float(status['success_rate']),
                                "consecutive_failures": int(status['consecutive_failures']),
                                "is_monitoring": bool(status['is_monitoring']),
                                "status": "Active" if status['is_monitoring'] else "Stopped"
                            }
                        )
                    except Exception as e:
                        print(f"⚠️  Failed to save session to database: {e}")
            
            # Configure row colors
            self.tree.tag_configure("good", foreground="#00ff88")
            self.tree.tag_configure("warning", foreground="#ffaa00")
            self.tree.tag_configure("danger", foreground="#ff4444")
            
        except Exception as e:
            self.log(f"Error updating display: {str(e)}", level="ERROR")
        
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
        self.log(f"Viewed details for {user_id}", user_id=user_id)
    
    def stop_selected_user(self):
        """Stop monitoring for selected user."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        user_id = item['values'][0]
        
        if user_id in AdminPanel.all_monitors:
            monitor = AdminPanel.all_monitors[user_id]
            status = monitor.get_monitoring_status()
            
            # Save final session data to database
            if db_instance:
                try:
                    db_instance.save_log(
                        level="INFO",
                        message=f"Monitoring session stopped by admin",
                        module="admin_panel_session",
                        user_id=user_id,
                        extra_data={
                            "total_checks": status['total_checks'],
                            "successful_checks": status['successful_checks'],
                            "failed_checks": status['failed_checks'],
                            "success_rate": status['success_rate'],
                            "stopped_by": "admin"
                        }
                    )
                except Exception as e:
                    print(f"⚠️  Failed to save session data: {e}")
            
            monitor.stop_monitoring()
            del AdminPanel.all_monitors[user_id]
            self.log(f"Stopped monitoring for {user_id}", user_id=user_id, level="WARNING")
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
    
    def log(self, message: str, level: str = "INFO", user_id: str = None):
        """Add message to activity log and save to database."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert("end", log_entry)
        self.log_text.see("end")
        
        # Save to database
        if db_instance:
            try:
                db_instance.save_log(
                    level=level,
                    message=message,
                    module="admin_panel",
                    user_id=user_id
                )
            except Exception as e:
                print(f"⚠️  Failed to log to database: {e}")
    
    def on_tab_changed(self, event):
        """Handle tab change events to refresh data."""
        try:
            selected_tab = event.widget.select()
            tab_text = event.widget.tab(selected_tab, "text")
            
            # Refresh data based on selected tab
            if "Database History" in tab_text:
                self.log("Refreshing database history...")
                self.load_database_logs()
            elif "Login History" in tab_text:
                self.log("Refreshing login history...")
                self.load_login_history()
            elif "Live Monitoring" in tab_text:
                self.log("Live monitoring view active")
        except Exception as e:
            print(f"⚠️  Error on tab change: {e}")
    
    def run(self):
        """Run the admin panel."""
        self.window.mainloop()
    
    @classmethod
    def add_monitor(cls, user_id: str, monitor: VoiceMonitor):
        """Add a monitor to the admin panel tracking."""
        cls.all_monitors[user_id] = monitor
        
        # Log session start to database
        if db_instance:
            try:
                db_instance.save_log(
                    level="INFO",
                    message=f"Monitoring session started",
                    module="admin_panel_session",
                    user_id=user_id,
                    extra_data={
                        "action": "session_started",
                        "threshold": monitor.threshold,
                        "check_interval": monitor.check_interval
                    }
                )
            except Exception as e:
                print(f"⚠️  Failed to log session start: {e}")
    
    @classmethod
    def remove_monitor(cls, user_id: str):
        """Remove a monitor from tracking."""
        if user_id in cls.all_monitors:
            # Save final session data if available
            if db_instance:
                try:
                    monitor = cls.all_monitors[user_id]
                    status = monitor.get_monitoring_status()
                    db_instance.save_log(
                        level="INFO",
                        message=f"Monitoring session ended",
                        module="admin_panel_session",
                        user_id=user_id,
                        extra_data={
                            "action": "session_ended",
                            "total_checks": status['total_checks'],
                            "successful_checks": status['successful_checks'],
                            "failed_checks": status['failed_checks'],
                            "success_rate": status['success_rate']
                        }
                    )
                except Exception as e:
                    print(f"⚠️  Failed to log session end: {e}")
            
            del cls.all_monitors[user_id]


if __name__ == "__main__":
    print("Starting Admin Panel...")
    print("Note: Start monitoring sessions from the main app first")
    print("Then open this admin panel to view them")
    
    panel = AdminPanel()
    panel.run()
