[Back to Index](../API.md)

# FileSystemService

The FileSystemService handles file system operations, data persistence, and caching for Sophia3.

## Core Features

### Data Management
- Message cache storage
- Configuration persistence
- Log management
- Temporary file handling

### File Operations
- Cache file operations
- Data serialization
- Directory management
- Path resolution

## Method Reference

### saveCache
```typescript
static async saveCache(
  channelId: string,
  data: CacheData,
  options?: SaveOptions
): Promise<void>
```

Saves message cache to disk.

#### Parameters:
- `channelId`: Channel identifier
- `data`: Cache data object
- `options`: Save settings
  - `compress`: Enable compression
  - `backup`: Create backup
  - `expiry`: Cache lifetime

### loadCache
```typescript
static async loadCache(
  channelId: string,
  options?: LoadOptions
): Promise<CacheData | null>
```

Loads message cache from disk.

#### Parameters:
- `channelId`: Channel identifier
- `options`: Load settings
  - `validateExpiry`: Check expiration
  - `fallback`: Default value
  - `decompress`: Auto-decompress

#### Returns:
Cached data or null if not found

### cleanupCache
```typescript
static async cleanupCache(
  options?: CleanupOptions
): Promise<void>
```

Performs cache cleanup.

#### Parameters:
- `options`: Cleanup settings
  - `older`: Age threshold
  - `size`: Size limit
  - `pattern`: File pattern

### ensureDirectory
```typescript
static async ensureDirectory(
  path: string,
  options?: DirectoryOptions
): Promise<void>
```

Creates directory if not exists.

#### Parameters:
- `path`: Directory path
- `options`: Directory settings
  - `mode`: Access permissions
  - `recursive`: Create parents
  - `clean`: Remove existing

## Integration Examples

### Cache Management
```typescript
// Save channel cache
await FileSystemService.saveCache(
  channel.id,
  messageCache,
  {
    compress: true,
    expiry: '1h'
  }
);

// Load cached data
const cache = await FileSystemService.loadCache(
  channel.id,
  { validateExpiry: true }
);
```

### Directory Management
```typescript
// Ensure cache directory
await FileSystemService.ensureDirectory(
  './data/cache',
  {
    recursive: true,
    clean: false
  }
);
```

### Cache Cleanup
```typescript
// Clean old cache files
await FileSystemService.cleanupCache({
  older: '24h',
  pattern: '*.cache.json'
});
```

## Error Handling

### File Operations
```typescript
try {
  await FileSystemService.saveCache(channelId, data);
} catch (error) {
  if (error instanceof FileSystemError) {
    console.error('Cache save failed:', error.message);
    await notifyAdmins(error);
  }
  throw error;
}
```

### Cache Loading
```typescript
try {
  const cache = await FileSystemService.loadCache(channelId);
} catch (error) {
  if (error instanceof CacheError) {
    console.warn('Using fallback cache:', error);
    return createEmptyCache();
  }
}
```

## Best Practices

1. **Data Management**
   - Regular cleanup
   - Proper error handling
   - Data validation

2. **Performance**
   - Use compression
   - Implement caching
   - Batch operations

3. **Security**
   - Validate paths
   - Handle permissions
   - Secure storage

## Configuration

```typescript
const FS_CONFIG = {
  // Path settings
  PATHS: {
    CACHE: './data/cache',
    LOGS: './logs',
    TEMP: './temp'
  },
  
  // Cache settings
  CACHE: {
    MAX_SIZE: '1GB',
    MAX_AGE: '24h',
    COMPRESSION: true
  },
  
  // Cleanup settings
  CLEANUP: {
    INTERVAL: '1h',
    MIN_AGE: '1h',
    BATCH_SIZE: 100
  },
  
  // Security settings
  SECURITY: {
    ALLOWED_PATHS: ['data', 'logs', 'temp'],
    FILE_PERMISSIONS: 0o644,
    DIR_PERMISSIONS: 0o755
  }
};
```

For implementation examples, see the [Examples Guide](../guides/Examples.md).