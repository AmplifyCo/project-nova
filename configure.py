#!/usr/bin/env python3
"""
Digital Twin Configuration Script
Interactive setup wizard for all bot credentials and tools.
Auto-detects provider settings and manages sensitive credentials.

Usage:
    python configure.py                    # Full setup wizard
    python configure.py --core-only        # Configure API keys only
    python configure.py --email-only       # Configure email only
    python configure.py --calendar-only    # Configure calendar only
    python configure.py --telegram-only    # Configure Telegram only
    python configure.py --tunnel-only      # Configure Cloudflare Tunnel only
"""

import os
import sys
import argparse
from pathlib import Path


# Email provider configurations
EMAIL_PROVIDERS = {
    'gmail.com': {
        'imap_server': 'imap.gmail.com',
        'smtp_server': 'smtp.gmail.com',
        'imap_port': 993,
        'smtp_port': 587,
        'caldav_url_template': 'https://apidata.googleusercontent.com/caldav/v2/{email}/events',
        'caldav_calendar_name': 'primary',
        'setup_url': 'https://myaccount.google.com/apppasswords',
        'setup_instructions': [
            '1. Enable 2FA: https://myaccount.google.com/security',
            '2. Create App Password: https://myaccount.google.com/apppasswords',
            '3. Use the 16-character app password (NOT your regular password)'
        ]
    },
    'googlemail.com': {
        'imap_server': 'imap.gmail.com',
        'smtp_server': 'smtp.gmail.com',
        'imap_port': 993,
        'smtp_port': 587,
        'caldav_url_template': 'https://apidata.googleusercontent.com/caldav/v2/{email}/events',
        'caldav_calendar_name': 'primary',
        'setup_url': 'https://myaccount.google.com/apppasswords',
        'setup_instructions': [
            '1. Enable 2FA: https://myaccount.google.com/security',
            '2. Create App Password: https://myaccount.google.com/apppasswords',
            '3. Use the 16-character app password (NOT your regular password)'
        ]
    },
    'outlook.com': {
        'imap_server': 'outlook.office365.com',
        'smtp_server': 'smtp.office365.com',
        'imap_port': 993,
        'smtp_port': 587,
        'caldav_url': 'https://outlook.office365.com/',
        'caldav_calendar_name': 'Calendar',
        'setup_url': 'https://account.microsoft.com/security',
        'setup_instructions': [
            '1. Enable 2FA: https://account.microsoft.com/security',
            '2. Create App Password in security settings',
            '3. Use the app password (NOT your regular password)'
        ]
    },
    'hotmail.com': {
        'imap_server': 'outlook.office365.com',
        'smtp_server': 'smtp.office365.com',
        'imap_port': 993,
        'smtp_port': 587,
        'caldav_url': 'https://outlook.office365.com/',
        'caldav_calendar_name': 'Calendar',
        'setup_url': 'https://account.microsoft.com/security',
        'setup_instructions': [
            '1. Enable 2FA: https://account.microsoft.com/security',
            '2. Create App Password in security settings',
            '3. Use the app password (NOT your regular password)'
        ]
    },
    'live.com': {
        'imap_server': 'outlook.office365.com',
        'smtp_server': 'smtp.office365.com',
        'imap_port': 993,
        'smtp_port': 587,
        'caldav_url': 'https://outlook.office365.com/',
        'caldav_calendar_name': 'Calendar',
        'setup_url': 'https://account.microsoft.com/security',
        'setup_instructions': [
            '1. Enable 2FA: https://account.microsoft.com/security',
            '2. Create App Password in security settings',
            '3. Use the app password (NOT your regular password)'
        ]
    },
    'yahoo.com': {
        'imap_server': 'imap.mail.yahoo.com',
        'smtp_server': 'smtp.mail.yahoo.com',
        'imap_port': 993,
        'smtp_port': 587,
        'setup_instructions': [
            '1. Enable 2FA in Yahoo account security',
            '2. Generate app password',
            '3. Use the app password'
        ]
    },
    'icloud.com': {
        'imap_server': 'imap.mail.me.com',
        'smtp_server': 'smtp.mail.me.com',
        'imap_port': 993,
        'smtp_port': 587,
        'caldav_url': 'https://caldav.icloud.com/',
        'caldav_calendar_name': 'Home',
        'setup_url': 'https://appleid.apple.com/account/manage',
        'setup_instructions': [
            '1. Go to https://appleid.apple.com/account/manage',
            '2. Generate app-specific password',
            '3. Use the app-specific password'
        ]
    }
}


class ConfigurationWizard:
    """Interactive configuration wizard for Digital Twin."""

    def __init__(self, env_path='.env'):
        self.env_path = Path(env_path)
        self.config = {}
        self.load_existing_config()

    def load_existing_config(self):
        """Load existing .env file if it exists."""
        if self.env_path.exists():
            with open(self.env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        self.config[key.strip()] = value.strip()

    def save_config(self):
        """Save configuration to .env file."""
        # Read existing file to preserve structure and comments
        lines = []
        if self.env_path.exists():
            with open(self.env_path, 'r') as f:
                lines = f.readlines()

        # Update or append configuration values
        updated_keys = set()
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and '=' in stripped:
                key = stripped.split('=', 1)[0].strip()
                if key in self.config:
                    new_lines.append(f"{key}={self.config[key]}\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Append new keys that weren't in the file
        if updated_keys != set(self.config.keys()):
            new_lines.append('\n# ==========================================\n')
            new_lines.append('# AUTO-GENERATED CONFIGURATION\n')
            new_lines.append('# ==========================================\n')
            for key, value in self.config.items():
                if key not in updated_keys:
                    new_lines.append(f'{key}={value}\n')

        # Write back
        with open(self.env_path, 'w') as f:
            f.writelines(new_lines)

        print(f"\n✅ Configuration saved to {self.env_path}")

    def print_header(self, text):
        """Print formatted header."""
        print("\n" + "=" * 60)
        print(f"  {text}")
        print("=" * 60)

    def get_input(self, prompt, default=None, required=True):
        """Get user input with optional default."""
        if default:
            prompt = f"{prompt} [{default}]"

        while True:
            value = input(f"{prompt}: ").strip()
            if value:
                return value
            if default:
                return default
            if not required:
                return None
            print("❌ This field is required. Please provide a value.")

    def get_password(self, prompt):
        """Get password input (no masking for simplicity)."""
        import getpass
        return getpass.getpass(f"{prompt}: ")

    def detect_provider(self, email):
        """Detect email provider from email address."""
        if '@' not in email:
            return None
        domain = email.split('@')[1].lower()
        return EMAIL_PROVIDERS.get(domain)

    def configure_core(self):
        """Configure core API credentials."""
        self.print_header("🔑 CORE CONFIGURATION")

        print("\nCore API credentials for the Digital Twin bot.")
        print("These are required for the bot to function.\n")

        # Anthropic API Key
        current_key = self.config.get('ANTHROPIC_API_KEY', '')
        if current_key and current_key != 'your-anthropic-api-key-here':
            print(f"Current Anthropic API Key: {current_key[:10]}...{current_key[-4:]}")
            update = self.get_input("Update Anthropic API key? (y/n)", default="n")
            if update.lower() not in ['y', 'yes']:
                print("✅ Keeping existing API key")
            else:
                api_key = self.get_input("Enter Anthropic API key")
                self.config['ANTHROPIC_API_KEY'] = api_key
        else:
            print("Get your API key from: https://console.anthropic.com/settings/keys\n")
            api_key = self.get_input("Enter Anthropic API key")
            self.config['ANTHROPIC_API_KEY'] = api_key

        print("\n✅ Core configuration complete!")

    def configure_telegram(self):
        """Configure Telegram bot credentials."""
        self.print_header("💬 TELEGRAM BOT CONFIGURATION")

        print("\nTelegram bot allows you to interact with your Digital Twin via Telegram.")
        print("Setup: Talk to @BotFather on Telegram to create a bot.\n")

        # Bot Token
        current_token = self.config.get('TELEGRAM_BOT_TOKEN', '')
        if current_token and current_token != '':
            print(f"Current Bot Token: {current_token[:10]}...{current_token[-4:]}")
            update = self.get_input("Update Telegram bot token? (y/n)", default="n")
            if update.lower() not in ['y', 'yes']:
                print("✅ Keeping existing bot token")
            else:
                token = self.get_input("Enter Telegram bot token")
                self.config['TELEGRAM_BOT_TOKEN'] = token
        else:
            print("📋 How to get bot token:")
            print("   1. Open Telegram and search for @BotFather")
            print("   2. Send /newbot command")
            print("   3. Follow instructions to create your bot")
            print("   4. Copy the bot token\n")

            token = self.get_input("Enter Telegram bot token (or leave empty to skip)", required=False)
            if token:
                self.config['TELEGRAM_BOT_TOKEN'] = token

        # Chat ID (optional)
        if 'TELEGRAM_BOT_TOKEN' in self.config and self.config['TELEGRAM_BOT_TOKEN']:
            print("\n📋 Chat ID (optional - for proactive notifications):")
            print("   1. Send /start to your bot")
            print("   2. Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates")
            print("   3. Look for 'chat':{'id': YOUR_CHAT_ID}")

            chat_id = self.get_input("\nEnter Telegram chat ID (leave empty to skip)", required=False)
            if chat_id:
                self.config['TELEGRAM_CHAT_ID'] = chat_id

        print("\n✅ Telegram configuration complete!")

    def configure_email(self):
        """Configure email settings."""
        self.print_header("📧 EMAIL CONFIGURATION")

        print("\nSupported providers (auto-configured):")
        print("  • Gmail (@gmail.com)")
        print("  • Outlook/Hotmail (@outlook.com, @hotmail.com, @live.com)")
        print("  • Yahoo (@yahoo.com)")
        print("  • iCloud (@icloud.com)")
        print("  • Custom IMAP/SMTP server\n")

        email = self.get_input("Enter your email address")
        provider = self.detect_provider(email)

        if provider:
            print(f"\n✅ Detected provider: {email.split('@')[1]}")
            print("\nSetup instructions:")
            for instruction in provider['setup_instructions']:
                print(f"   {instruction}")

            if 'setup_url' in provider:
                print(f"\n🔗 Setup URL: {provider['setup_url']}")

            # Auto-populate settings
            self.config['EMAIL_ADDRESS'] = email
            self.config['EMAIL_IMAP_SERVER'] = provider['imap_server']
            self.config['EMAIL_SMTP_SERVER'] = provider['smtp_server']
            self.config['EMAIL_IMAP_PORT'] = str(provider['imap_port'])
            self.config['EMAIL_SMTP_PORT'] = str(provider['smtp_port'])

            print(f"\n📋 Auto-configured:")
            print(f"   IMAP Server: {provider['imap_server']}:{provider['imap_port']}")
            print(f"   SMTP Server: {provider['smtp_server']}:{provider['smtp_port']}")

            password = self.get_password("\nEnter app password (input hidden)")
            self.config['EMAIL_PASSWORD'] = password

        else:
            print("\n⚙️  Custom email server configuration")
            self.config['EMAIL_ADDRESS'] = email
            self.config['EMAIL_IMAP_SERVER'] = self.get_input("IMAP server")
            self.config['EMAIL_IMAP_PORT'] = self.get_input("IMAP port", default="993")
            self.config['EMAIL_SMTP_SERVER'] = self.get_input("SMTP server")
            self.config['EMAIL_SMTP_PORT'] = self.get_input("SMTP port", default="587")
            password = self.get_password("Email password")
            self.config['EMAIL_PASSWORD'] = password

        print("\n✅ Email configuration complete!")

    def configure_calendar(self):
        """Configure calendar settings."""
        self.print_header("📅 CALENDAR CONFIGURATION")

        # Check if email is already configured
        email = self.config.get('EMAIL_ADDRESS')
        if not email:
            email = self.get_input("Enter your email address (for calendar)")

        provider = self.detect_provider(email)

        if provider and ('caldav_url' in provider or 'caldav_url_template' in provider):
            print(f"\n✅ Calendar detected for: {email.split('@')[1]}")

            # Auto-populate CalDAV settings
            if 'caldav_url_template' in provider:
                caldav_url = provider['caldav_url_template'].format(email=email)
            else:
                caldav_url = provider['caldav_url']

            self.config['CALDAV_URL'] = caldav_url
            self.config['CALDAV_USERNAME'] = email
            self.config['CALDAV_CALENDAR_NAME'] = provider.get('caldav_calendar_name', 'primary')

            print(f"\n📋 Auto-configured:")
            print(f"   CalDAV URL: {caldav_url}")
            print(f"   Calendar: {provider.get('caldav_calendar_name', 'primary')}")

            # Reuse email password if available
            if 'EMAIL_PASSWORD' in self.config:
                reuse = self.get_input("\nUse same app password as email? (y/n)", default="y")
                if reuse.lower() in ['y', 'yes']:
                    self.config['CALDAV_PASSWORD'] = self.config['EMAIL_PASSWORD']
                else:
                    password = self.get_password("Enter CalDAV password")
                    self.config['CALDAV_PASSWORD'] = password
            else:
                password = self.get_password("\nEnter CalDAV password (app password)")
                self.config['CALDAV_PASSWORD'] = password

        else:
            print("\n⚙️  Custom CalDAV server configuration")
            self.config['CALDAV_URL'] = self.get_input("CalDAV server URL")
            self.config['CALDAV_USERNAME'] = self.get_input("CalDAV username", default=email)
            password = self.get_password("CalDAV password")
            self.config['CALDAV_PASSWORD'] = password
            self.config['CALDAV_CALENDAR_NAME'] = self.get_input(
                "Calendar name",
                default="primary",
                required=False
            ) or "primary"

        print("\n✅ Calendar configuration complete!")

    def configure_cloudflare_tunnel(self):
        """Configure Cloudflare Tunnel settings."""
        self.print_header("🌐 CLOUDFLARE TUNNEL CONFIGURATION")

        print("\nCloudflare Tunnel exposes your Digital Twin bot to the internet securely.")
        print("Perfect for webhooks, remote access, or sharing with others.\n")

        print("📋 How to get your tunnel token:")
        print("   1. Go to: https://one.dash.cloudflare.com/")
        print("   2. Navigate to: Zero Trust > Networks > Tunnels")
        print("   3. Create a new tunnel (or select existing)")
        print("   4. Copy the tunnel token from the setup page")
        print("   5. Paste it here\n")

        # Check if tunnel token already exists
        current_token = self.config.get('CLOUDFLARE_TUNNEL_TOKEN', '')
        if current_token:
            print(f"Current Tunnel Token: {current_token[:20]}...{current_token[-10:]}")
            update = self.get_input("Update Cloudflare tunnel token? (y/n)", default="n")
            if update.lower() not in ['y', 'yes']:
                print("✅ Keeping existing tunnel token")
                return

        # Get tunnel token
        tunnel_token = self.get_input("Enter Cloudflare tunnel token (or leave empty to skip)", required=False)

        if not tunnel_token:
            print("\n⏭️  Skipping Cloudflare Tunnel configuration")
            return

        self.config['CLOUDFLARE_TUNNEL_TOKEN'] = tunnel_token

        # Optional: Local service URL
        print("\n📋 Local service configuration:")
        print("   What local service should the tunnel expose?")
        print("   Examples:")
        print("     • http://localhost:8000 (web interface)")
        print("     • http://localhost:5000 (API server)")
        print("     • Leave empty to configure manually later\n")

        local_url = self.get_input(
            "Local service URL",
            default="http://localhost:8000",
            required=False
        )

        if local_url:
            self.config['CLOUDFLARE_TUNNEL_LOCAL_URL'] = local_url
            print(f"\n✅ Tunnel will expose: {local_url}")

        print("\n✅ Cloudflare Tunnel configuration complete!")
        print("\n💡 Note: Run 'dt-setup tunnel start' to start the tunnel after setup")

    def run(self, core_only=False, telegram_only=False, email_only=False, calendar_only=False, tunnel_only=False):
        """Run the configuration wizard."""
        print("\n" + "=" * 60)
        print("  🤖 Digital Twin Configuration Wizard")
        print("=" * 60)
        print("\nThis wizard will help you set up:")
        print("  • Core API credentials (Anthropic)")
        print("  • Communication tools (Telegram, Email, Calendar)")
        print("  • Cloudflare Tunnel (optional)")
        print("  • Sensitive passwords and tokens")
        print(f"\nConfiguration will be saved to: {self.env_path}")

        # Core configuration (API keys)
        if not any([telegram_only, email_only, calendar_only, tunnel_only]):
            setup_core = self.get_input("\nConfigure core API keys? (y/n)", default="y")
            if setup_core.lower() in ['y', 'yes']:
                self.configure_core()

        # Telegram configuration
        if not any([core_only, email_only, calendar_only, tunnel_only]):
            setup_telegram = self.get_input("\nConfigure Telegram bot? (y/n)", default="n")
            if setup_telegram.lower() in ['y', 'yes']:
                self.configure_telegram()

        # Email configuration
        if not any([core_only, telegram_only, calendar_only, tunnel_only]):
            setup_email = self.get_input("\nConfigure email? (y/n)", default="n")
            if setup_email.lower() in ['y', 'yes']:
                self.configure_email()

        # Calendar configuration
        if not any([core_only, telegram_only, email_only, tunnel_only]):
            setup_calendar = self.get_input("\nConfigure calendar? (y/n)", default="n")
            if setup_calendar.lower() in ['y', 'yes']:
                self.configure_calendar()

        # Cloudflare Tunnel configuration
        if not any([core_only, telegram_only, email_only, calendar_only]):
            setup_tunnel = self.get_input("\nConfigure Cloudflare Tunnel? (y/n)", default="n")
            if setup_tunnel.lower() in ['y', 'yes']:
                self.configure_cloudflare_tunnel()

        # Save configuration
        self.save_config()

        # Next steps
        print("\n" + "=" * 60)
        print("  🎉 Setup Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Verify your .env file contains the correct settings")
        print("  2. Restart your Digital Twin bot")
        print("  3. Check logs for:")
        print("     • 📧 Email tool registered (if configured)")
        print("     • 📅 Calendar tool registered (if configured)")
        print("     • 💬 Telegram bot started (if configured)")
        if 'CLOUDFLARE_TUNNEL_TOKEN' in self.config:
            print("     • 🌐 Cloudflare Tunnel (run 'dt-setup tunnel start')")
        print("\nTest your configuration:")
        print("  • Telegram: Send /start to your bot")
        print("  • Email: 'Check my emails'")
        print("  • Calendar: 'What's on my calendar today?'")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Digital Twin Configuration Wizard - Manage API keys and communication tools',
        epilog='Examples:\n'
               '  python configure.py              # Full setup wizard\n'
               '  python configure.py --core-only  # Configure API keys only\n'
               '  python configure.py --email-only # Configure email only\n'
               '  python configure.py --tunnel-only # Configure Cloudflare Tunnel only\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--core-only',
        action='store_true',
        help='Configure core API keys only (Anthropic)'
    )
    parser.add_argument(
        '--telegram-only',
        action='store_true',
        help='Configure Telegram bot only'
    )
    parser.add_argument(
        '--email-only',
        action='store_true',
        help='Configure email only'
    )
    parser.add_argument(
        '--calendar-only',
        action='store_true',
        help='Configure calendar only'
    )
    parser.add_argument(
        '--tunnel-only',
        action='store_true',
        help='Configure Cloudflare Tunnel only'
    )
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file (default: .env)'
    )
    parser.add_argument(
        '--talents',
        action='store_true',
        help='Show all talents and their configuration status'
    )

    args = parser.parse_args()

    if args.talents:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.core.talents.catalog import TalentCatalog
        TalentCatalog().print_status()
        return

    wizard = ConfigurationWizard(env_path=args.env_file)
    wizard.run(
        core_only=args.core_only,
        telegram_only=args.telegram_only,
        email_only=args.email_only,
        calendar_only=args.calendar_only,
        tunnel_only=args.tunnel_only
    )


if __name__ == '__main__':
    main()
