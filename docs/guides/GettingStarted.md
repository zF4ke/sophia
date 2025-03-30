[Back to Index](../API.md)

# Getting Started with Sophia3

## Prerequisites

- Node.js 18.0.0 or higher
- Discord server with admin permissions
- Google Cloud account for Gemini API
- Basic understanding of Discord slash commands

## Installation

1. **Clone Repository**
```bash
git clone https://github.com/your-username/sophia3.git
cd sophia3
```

2. **Install Dependencies**
```bash
npm install
```

3. **Configure Environment**
Create a `.env` file in the root directory:
```env
DISCORD_TOKEN=your_discord_bot_token
GEMINI_API_KEY=your_google_cloud_api_key
```

4. **Build Project**
```bash
npm run build
```

## Initial Setup

### Bot Setup
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Add a bot user
4. Enable required intents:
   - Message Content
   - Server Members
   - Guild Messages

### Server Configuration
1. Invite bot to your server using OAuth2 URL with scopes:
   - `applications.commands`
   - `bot`
2. Required bot permissions:
   - Read Messages
   - Send Messages
   - Manage Messages
   - Read Message History
   - Add Reactions
   - Use Slash Commands

### First Run
1. Start the bot:
```bash
npm run start
```

2. Verify bot is online:
```
/ping
```

## Basic Usage

### Search Feature
```
/search topic:"meeting notes" channel:#team-updates
```
- Searches for conversations about specified topic
- Shows relevance-scored results
- Supports pagination and filtering

### Context Analysis
```
/context prompt:"What was decided?" channel:#project
```
- Gets AI-powered responses based on channel context
- Understands conversation history
- Provides relevant summaries

### Message Retrieval
```
/getmessage channel:#announcements number:50
```
- Gets specific messages with context
- Shows surrounding conversation
- Supports private viewing

## Access Control

### Setting Up Roles
1. Create roles in Discord:
   - Bot Admin
   - Bot Moderator
   - Bot User

2. Configure access:
```
/access action:grant command:search role:@Bot Moderator
```

### Managing Permissions
- Grant command access to roles
- Set access durations
- Restrict to specific channels
- Monitor usage with rate limits

## Performance Optimization

### Caching
1. Enable channel caching:
```
/cache action:status channel:#general
```

2. Regular maintenance:
```
/cache action:clean
```

### Best Practices
- Cache active channels
- Clean old caches regularly
- Monitor storage usage
- Update cache after major changes

## Common Operations

### Finding Information
1. Search for topic:
```
/search topic:"project deadlines" channel:#project
```

2. Get context:
```
/context prompt:"Summarize recent updates" channel:#project
```

### Managing Results
- Use pagination controls
- Apply result filters
- Save important findings
- Share results when needed

## Troubleshooting

### Common Issues
1. **Rate Limits**
   - Wait for cooldown
   - Use ephemeral responses
   - Cache frequent searches

2. **Permission Errors**
   - Verify role setup
   - Check channel permissions
   - Review access settings

3. **Cache Issues**
   - Clear problematic cache
   - Rebuild if necessary
   - Monitor storage space

### Getting Help
- Check error messages
- Review documentation
- Contact support team
- Monitor bot status

## Next Steps

1. **Explore Features**
   - Try all basic commands
   - Test advanced options
   - Understand limits

2. **Configure Bot**
   - Set up access roles
   - Configure caching
   - Customize responses

3. **Train Users**
   - Share documentation
   - Demonstrate features
   - Establish guidelines

For detailed API documentation, see the [API Reference](../API.md).
For implementation examples, see the [Examples Guide](Examples.md).