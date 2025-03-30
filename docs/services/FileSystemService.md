[Back to Index](../API.md)

# FileSystemService

The FileSystemService centralizes all file system operations within the application, providing a unified interface for file I/O operations. It handles reading, writing, and managing files in a consistent manner, with specialized support for JSON data, service-specific storage, and different file types.

## Properties

### Directory Constants

```typescript
private static readonly BASE_DATA_DIR = path.join(process.cwd(), 'data');
private static readonly CACHE_DIR = path.join(this.BASE_DATA_DIR, 'cache');
```

These properties define the standard directory structure:
- `BASE_DATA_DIR`: The root directory for all persistent data
- `CACHE_DIR`: Subdirectory specifically for cached data

## Core File Methods

### `ensureDirectoryExists`

```typescript
public static ensureDirectoryExists(dirPath: string): void
```

Creates a directory if it doesn't already exist.

#### Features:
- Simplifies directory creation throughout the application
- Creates parent directories as needed
- Safe to call on existing directories

#### Parameters:
- `dirPath: string` - Full path to the directory to ensure exists

#### Example:
```typescript
FileSystemService.ensureDirectoryExists('./data/custom');
```

### `fileExists`

```typescript
public static fileExists(filePath: string): boolean
```

Checks if a file exists.

#### Features:
- Simple utility for checking file existence
- Provides clean abstraction over fs.existsSync

#### Parameters:
- `filePath: string` - Path to the file to check

#### Returns:
- `boolean` - True if file exists, false otherwise

#### Example:
```typescript
if (FileSystemService.fileExists('./data/config.json')) {
  // File exists, proceed
}
```

### `getFileStats`

```typescript
public static getFileStats(filePath: string): fs.Stats | null
```

Gets file statistics (size, modification date, etc.).

#### Features:
- Provides metadata about files
- Handles non-existent files gracefully

#### Parameters:
- `filePath: string` - Path to the file

#### Returns:
- `fs.Stats | null` - File stats object or null if file doesn't exist

#### Example:
```typescript
const stats = FileSystemService.getFileStats('./data/config.json');
if (stats) {
  console.log(`File size: ${stats.size} bytes`);
  console.log(`Last modified: ${stats.mtime}`);
}
```

## JSON File Methods

### `readJsonFile`

```typescript
public static readJsonFile<T>(filePath: string): T | null
```

Reads and parses a JSON file with type safety.

#### Features:
- Strongly typed return value
- Handles non-existent files gracefully
- Provides error logging for invalid JSON

#### Parameters:
- `filePath: string` - Full path to the JSON file

#### Returns:
- `T | null` - Parsed JSON data with requested type, or null if the file doesn't exist or is invalid

#### Example:
```typescript
const config = FileSystemService.readJsonFile<AppConfig>('./data/config.json');
if (config) {
  // Use the config data
}
```

### `writeJsonFile`

```typescript
public static writeJsonFile(filePath: string, data: any): boolean
```

Writes data to a JSON file, creating directories as needed.

#### Features:
- Automatically creates parent directories
- Handles serialization to JSON with pretty formatting
- Reports success or failure

#### Parameters:
- `filePath: string` - Full path to the JSON file to write
- `data: any` - Data to serialize and write to the file

#### Returns:
- `boolean` - True if write was successful, false otherwise

#### Example:
```typescript
const success = FileSystemService.writeJsonFile('./data/settings.json', userSettings);
```

## Directory Operations

### `listFiles`

```typescript
public static listFiles(dirPath: string, extension?: string): string[]
```

Lists all files in a directory, optionally filtered by extension.

#### Features:
- Optional filtering by file extension
- Handles non-existent directories gracefully
- Returns empty array instead of throwing on errors

#### Parameters:
- `dirPath: string` - Directory to list files from
- `extension?: string` - Optional file extension to filter by (e.g., '.json')

#### Returns:
- `string[]` - Array of filenames (not full paths)

#### Example:
```typescript
const jsonFiles = FileSystemService.listFiles('./data', '.json');
```

### `deleteFile`

```typescript
public static deleteFile(filePath: string): boolean
```

Deletes a file if it exists.

#### Features:
- Safe deletion (checks existence first)
- Reports success or failure
- Clean error handling

#### Parameters:
- `filePath: string` - Full path to the file to delete

#### Returns:
- `boolean` - True if delete was successful or the file didn't exist, false on error

#### Example:
```typescript
FileSystemService.deleteFile('./data/temp.json');
```

## Cache Operations

### `getCacheFilePath`

```typescript
public static getCacheFilePath(entityId: string): string
```

Gets the standardized path to a cache file for a specific entity.

#### Features:
- Ensures consistent cache file location
- Creates cache directory if needed
- Enforces JSON file extension

#### Parameters:
- `entityId: string` - ID of the entity (e.g., channel ID)

#### Returns:
- `string` - Full path to the cache file

#### Example:
```typescript
const path = FileSystemService.getCacheFilePath('channel12345');
// Returns something like '/path/to/data/cache/channel12345.json'
```

### `readCache`

```typescript
public static readCache<T>(entityId: string): T | null
```

Reads cached entity data with type safety.

#### Features:
- Strongly typed return value
- Automatically resolves correct file path
- Returns null for non-existent or invalid cache files

#### Parameters:
- `entityId: string` - ID of the entity to read

#### Returns:
- `T | null` - Parsed cache data with requested type, or null if not found

#### Example:
```typescript
const channelCache = FileSystemService.readCache<ChannelData>('channel12345');
```

### `writeCache`

```typescript
public static writeCache(entityId: string, data: any): boolean
```

Writes entity data to the cache.

#### Features:
- Automatically resolves correct file path
- Creates cache directory if needed
- Reports success or failure

#### Parameters:
- `entityId: string` - ID of the entity to write
- `data: any` - Data to cache

#### Returns:
- `boolean` - True if write was successful, false otherwise

#### Example:
```typescript
FileSystemService.writeCache('channel12345', channelData);
```

### `clearCache`

```typescript
public static clearCache(entityId: string): boolean
```

Clears a specific entity from the cache.

#### Features:
- Automatically resolves correct file path
- Safe deletion (handles non-existent files)
- Reports success or failure

#### Parameters:
- `entityId: string` - ID of the entity to clear

#### Returns:
- `boolean` - True if delete was successful, false on error

#### Example:
```typescript
FileSystemService.clearCache('channel12345');
```

### `clearAllCaches`

```typescript
public static clearAllCaches(): boolean
```

Clears all entity caches.

#### Features:
- Removes all cache files
- Filters by JSON extension
- Reports success or failure

#### Returns:
- `boolean` - True if clear was successful, false on error

#### Example:
```typescript
FileSystemService.clearAllCaches();
```

## Service-Specific Storage

### `getServiceDataPath`

```typescript
public static getServiceDataPath(serviceName: string, fileName: string): string
```

Gets the path for a service-specific data file.

#### Features:
- Standardizes data organization by service
- Creates service directory if needed
- Allows services to maintain isolated data files

#### Parameters:
- `serviceName: string` - Name of the service (e.g., 'security', 'ai', etc.)
- `fileName: string` - Name of the file within the service's data directory

#### Returns:
- `string` - Full path to the service data file

#### Example:
```typescript
const configPath = FileSystemService.getServiceDataPath('security', 'admins.json');
```

### `readServiceData`

```typescript
public static readServiceData<T>(serviceName: string, fileName: string): T | null
```

Reads a service-specific data file with type safety.

#### Features:
- Typed return values for service-specific data
- Automatic path resolution
- Consistent error handling

#### Parameters:
- `serviceName: string` - Name of the service
- `fileName: string` - Name of the file within the service's data directory

#### Returns:
- `T | null` - Parsed data with requested type, or null if not found

#### Example:
```typescript
const admins = FileSystemService.readServiceData<string[]>('security', 'admins.json');
```

### `writeServiceData`

```typescript
public static writeServiceData(serviceName: string, fileName: string, data: any): boolean
```

Writes data to a service-specific file.

#### Features:
- Automatic path resolution
- Creates service directories as needed
- Consistent error handling

#### Parameters:
- `serviceName: string` - Name of the service
- `fileName: string` - Name of the file within the service's data directory
- `data: any` - Data to write

#### Returns:
- `boolean` - True if write was successful, false otherwise

#### Example:
```typescript
FileSystemService.writeServiceData('security', 'admins.json', ['user1', 'user2']);
```

### `listServiceFiles`

```typescript
public static listServiceFiles(serviceName: string, extension?: string): string[]
```

Lists all files in a service's data directory.

#### Features:
- Optional filtering by extension
- Consistent error handling
- Automatic service directory resolution

#### Parameters:
- `serviceName: string` - Name of the service
- `extension?: string` - Optional file extension filter

#### Returns:
- `string[]` - Array of file names in the service's directory

#### Example:
```typescript
const configFiles = FileSystemService.listServiceFiles('security', '.json');
```

### `deleteServiceFile`

```typescript
public static deleteServiceFile(serviceName: string, fileName: string): boolean
```

Deletes a service-specific data file.

#### Features:
- Automatic path resolution
- Safe deletion with existence check
- Consistent error handling

#### Parameters:
- `serviceName: string` - Name of the service
- `fileName: string` - Name of the file to delete

#### Returns:
- `boolean` - True if delete was successful or file didn't exist, false on error

#### Example:
```typescript
FileSystemService.deleteServiceFile('security', 'temporary.json');
```

## Text and Binary File Operations

### `readTextFile`

```typescript
public static readTextFile(filePath: string): string | null
```

Reads raw text file contents without JSON parsing.

#### Features:
- Reads any text-based file format
- Handles non-existent files gracefully
- Useful for configuration, templates, etc.

#### Parameters:
- `filePath: string` - Path to the file

#### Returns:
- `string | null` - File contents as string or null if file doesn't exist

#### Example:
```typescript
const template = FileSystemService.readTextFile('./templates/email.html');
```

### `writeTextFile`

```typescript
public static writeTextFile(filePath: string, content: string): boolean
```

Writes raw text content to a file.

#### Features:
- Creates directories as needed
- Works with any text-based format
- Consistent error handling

#### Parameters:
- `filePath: string` - Path to the file
- `content: string` - Text content to write

#### Returns:
- `boolean` - True if write was successful, false otherwise

#### Example:
```typescript
FileSystemService.writeTextFile('./logs/activity.log', logContent);
```

### `readBinaryFile`

```typescript
public static readBinaryFile(filePath: string): Buffer | null
```

Reads binary file contents.

#### Features:
- Supports image, audio, and other binary formats
- Handles non-existent files gracefully
- Returns Node.js Buffer for flexibility

#### Parameters:
- `filePath: string` - Path to the file

#### Returns:
- `Buffer | null` - File contents as Buffer or null if file doesn't exist

#### Example:
```typescript
const imageData = FileSystemService.readBinaryFile('./assets/logo.png');
```

### `writeBinaryFile`

```typescript
public static writeBinaryFile(filePath: string, content: Buffer): boolean
```

Writes binary content to a file.

#### Features:
- Creates directories as needed
- Works with any binary format
- Consistent error handling

#### Parameters:
- `filePath: string` - Path to the file
- `content: Buffer` - Binary content to write

#### Returns:
- `boolean` - True if write was successful, false otherwise

#### Example:
```typescript
FileSystemService.writeBinaryFile('./uploads/image.jpg', imageBuffer);
```

## Best Practices

1. Always use FileSystemService for file operations rather than direct fs calls
2. Leverage type parameters with readJsonFile, readCache, and readServiceData for type safety
3. Use service-specific storage methods to isolate each service's data
4. Use the path utilities to ensure consistent file locations
5. Check return values from write and delete operations to handle errors appropriately
6. Use the cache-specific methods for all cache operations to maintain consistency
7. Consider using text and binary file operations for specialized file formats