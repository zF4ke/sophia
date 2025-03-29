[Back to Index](../API.md)

# ConversationService

The ConversationService is responsible for organizing Discord messages into meaningful conversation groups based on time gaps and message relationships. It provides sophisticated conversation grouping and filtering capabilities.

## Properties

### Time Constants

```typescript
private static readonly MAX_CONVERSATION_GAP = 5 * 60 * 1000;   // 5 minutes
private static readonly SAME_AUTHOR_GAP = 8 * 60 * 1000;        // 8 minutes
private static readonly REPLY_GAP = 10 * 60 * 1000;             // 10 minutes
```

These constants define the timing rules for conversation grouping:
- Standard gap between messages: 5 minutes
- Extended gap for same author: 8 minutes
- Extended gap for direct replies: 10 minutes

## Methods

### `groupMessagesByConversation`

```typescript
static groupMessagesByConversation(messages: Message[]): Message[][]
```

Groups messages into conversations based on time gaps and relationships between messages.

#### Features:
- Identifies conversation boundaries using smart timing rules
- Considers message relationships (replies, mentions)
- Handles continuous conversation threads intelligently
- Processes messages in reverse chronological order (newest to oldest)

#### Parameters:
- `messages: Message[]` - Array of messages to group into conversations

#### Returns:
- `Message[][]` - Array of conversation groups (each group is an array of messages)

#### Example:
```typescript
const messages = await MessageService.fetchMessages(channel, 1000, interaction);
const conversations = ConversationService.groupMessagesByConversation(messages);
console.log(`Identified ${conversations.length} separate conversations`);
```

### `isRelatedToLastMessage` (Private)

```typescript
private static isRelatedToLastMessage(message: Message, currentConversation: Message[]): boolean
```

Determines if a message is part of the current conversation based on timing and relationship rules.

#### Features:
- Checks for direct replies to previous messages in conversation
- Extends conversation window for messages by the same author
- Applies standard timing window for regular messages

#### Parameters:
- `message: Message` - The message to evaluate
- `currentConversation: Message[]` - The current conversation group being built

#### Returns:
- `boolean` - True if the message belongs to the current conversation, false otherwise

### `filterValidConversations`

```typescript
static filterValidConversations(conversations: Message[][], includeBots: boolean = false): Message[][]
```

Filters out invalid conversations such as those that are too short or only contain bot messages.

#### Features:
- Removes single short messages that don't constitute meaningful conversations
- Option to include or exclude bot-only conversations
- Ensures high-quality conversation groups

#### Parameters:
- `conversations: Message[][]` - Array of conversation groups to filter
- `includeBots: boolean` - Whether to include bot-only conversations (default: false)

#### Returns:
- `Message[][]` - Filtered array of valid conversation groups

#### Example:
```typescript
const messages = await MessageService.fetchMessages(channel, 1000, interaction);
const conversations = ConversationService.groupMessagesByConversation(messages);
const validConversations = ConversationService.filterValidConversations(conversations);
console.log(`Found ${validConversations.length} valid conversations out of ${conversations.length}`);
```

## Usage Recommendations

- Use this service after fetching messages with MessageService
- Consider including bot messages in educational or announcement channels
- Filter valid conversations before performing AI analysis to improve efficiency and accuracy
- Fine-tune conversation grouping by adjusting the time constants if needed for specific channel dynamics