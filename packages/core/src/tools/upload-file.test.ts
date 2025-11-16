/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { UploadFileToolParams } from './upload-file.js';
import { UploadFileTool } from './upload-file.js';
import { ToolErrorType } from './tool-error.js';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import type { Config } from '../config/config.js';
import { FileDiscoveryService } from '../services/fileDiscoveryService.js';
import { StandardFileSystemService } from '../services/fileSystemService.js';
import { createMockWorkspaceContext } from '../test-utils/mockWorkspaceContext.js';
import { AuthType } from '../core/contentGenerator.js';

vi.mock('../telemetry/loggers.js', () => ({
  logFileOperation: vi.fn(),
}));

// Mock the @google/genai module
const { mockUpload, mockFilesApi } = vi.hoisted(() => {
  const mockUpload = vi.fn();
  const mockFilesApi = {
    upload: mockUpload,
  };
  return { mockUpload, mockFilesApi };
});

vi.mock('@google/genai', () => ({
  GoogleGenAI: vi.fn().mockImplementation(() => ({
    files: mockFilesApi,
  })),
}));

describe('UploadFileTool', () => {
  let tempRootDir: string;
  let tool: UploadFileTool;
  const abortSignal = new AbortController().signal;

  beforeEach(async () => {
    vi.resetAllMocks();
    
    // Create a unique temporary root directory for each test run
    tempRootDir = await fsp.mkdtemp(
      path.join(os.tmpdir(), 'upload-file-tool-root-'),
    );

    const mockConfigInstance = {
      getFileService: () => new FileDiscoveryService(tempRootDir),
      getFileSystemService: () => new StandardFileSystemService(),
      getTargetDir: () => tempRootDir,
      getWorkspaceContext: () => createMockWorkspaceContext(tempRootDir),
      storage: {
        getProjectTempDir: () => path.join(tempRootDir, '.temp'),
      },
      getContentGeneratorConfig: () => ({
        apiKey: 'test-api-key',
        vertexai: false,
        authType: AuthType.USE_GEMINI,
      }),
    } as unknown as Config;
    tool = new UploadFileTool(mockConfigInstance);
  });

  afterEach(async () => {
    // Clean up the temporary root directory
    if (fs.existsSync(tempRootDir)) {
      await fsp.rm(tempRootDir, { recursive: true, force: true });
    }
  });

  describe('build', () => {
    it('should return an invocation for valid params with existing file', async () => {
      const testFilePath = path.join(tempRootDir, 'test.txt');
      await fsp.writeFile(testFilePath, 'test content');
      
      const params: UploadFileToolParams = {
        absolute_path: testFilePath,
      };
      const result = tool.build(params);
      expect(typeof result).not.toBe('string');
    });

    it('should throw error if file path is relative', () => {
      const params: UploadFileToolParams = {
        absolute_path: 'relative/path.txt',
      };
      expect(() => tool.build(params)).toThrow(
        'File path must be absolute, but was relative: relative/path.txt. You must provide an absolute path.',
      );
    });

    it('should throw error if path is outside workspace', () => {
      const params: UploadFileToolParams = {
        absolute_path: '/outside/root.txt',
      };
      expect(() => tool.build(params)).toThrow(
        /File path must be within one of the workspace directories/,
      );
    });

    it('should throw error if file does not exist', () => {
      const params: UploadFileToolParams = {
        absolute_path: path.join(tempRootDir, 'nonexistent.txt'),
      };
      expect(() => tool.build(params)).toThrow(
        /File does not exist/,
      );
    });

    it('should throw error if path is a directory', async () => {
      const dirPath = path.join(tempRootDir, 'testdir');
      await fsp.mkdir(dirPath);
      
      const params: UploadFileToolParams = {
        absolute_path: dirPath,
      };
      expect(() => tool.build(params)).toThrow(
        /Path is not a file/,
      );
    });

    it('should throw error when using Vertex AI', async () => {
      const testFilePath = path.join(tempRootDir, 'test.txt');
      await fsp.writeFile(testFilePath, 'test content');

      const mockConfigInstance = {
        getFileService: () => new FileDiscoveryService(tempRootDir),
        getFileSystemService: () => new StandardFileSystemService(),
        getTargetDir: () => tempRootDir,
        getWorkspaceContext: () => createMockWorkspaceContext(tempRootDir),
        storage: {
          getProjectTempDir: () => path.join(tempRootDir, '.temp'),
        },
        getContentGeneratorConfig: () => ({
          apiKey: 'test-api-key',
          vertexai: true,
          authType: AuthType.USE_VERTEX_AI,
        }),
      } as unknown as Config;
      const vertexTool = new UploadFileTool(mockConfigInstance);
      
      const params: UploadFileToolParams = {
        absolute_path: testFilePath,
      };
      expect(() => vertexTool.build(params)).toThrow(
        /File upload is not supported when using Vertex AI/,
      );
    });

    it('should allow access to files in project temp directory', async () => {
      const tempDir = path.join(tempRootDir, '.temp');
      await fsp.mkdir(tempDir, { recursive: true });
      const testFilePath = path.join(tempDir, 'temp-file.txt');
      await fsp.writeFile(testFilePath, 'temp content');
      
      const params: UploadFileToolParams = {
        absolute_path: testFilePath,
      };
      const result = tool.build(params);
      expect(typeof result).not.toBe('string');
    });

    it('should throw error if path is empty', () => {
      const params: UploadFileToolParams = {
        absolute_path: '',
      };
      expect(() => tool.build(params)).toThrow(
        "The 'absolute_path' parameter must be non-empty.",
      );
    });
  });

  describe.skip('execute', () => {
    it('should upload file successfully', async () => {
      const mockUploadedFile = {
        name: 'files/test123',
        uri: 'https://generativelanguage.googleapis.com/v1beta/files/test123',
        displayName: 'test.txt',
        mimeType: 'text/plain',
        sizeBytes: 12,
        state: 'ACTIVE',
      };
      
      // Mock the upload method
      mockUpload.mockResolvedValue(mockUploadedFile);

      const testFilePath = path.join(tempRootDir, 'test.txt');
      await fsp.writeFile(testFilePath, 'test content');
      
      const params: UploadFileToolParams = {
        absolute_path: testFilePath,
        display_name: 'My Test File',
      };
      
      const invocation = tool.build(params);
      const result = await invocation.execute(abortSignal);
      
      expect(result.error).toBeUndefined();
      expect(result.llmContent).toContain('Successfully uploaded file');
      expect(result.llmContent).toContain('test.txt');
      expect(mockUpload).toHaveBeenCalledWith({
        file: testFilePath,
        config: {
          abortSignal,
          displayName: 'My Test File',
        },
      });
    });

    it('should handle upload errors', async () => {
      mockUpload.mockRejectedValue(new Error('Upload failed'));

      const testFilePath = path.join(tempRootDir, 'test.txt');
      await fsp.writeFile(testFilePath, 'test content');
      
      const params: UploadFileToolParams = {
        absolute_path: testFilePath,
      };
      
      const invocation = tool.build(params);
      const result = await invocation.execute(abortSignal);
      
      expect(result.error).toBeDefined();
      expect(result.error?.type).toBe(ToolErrorType.FILE_UPLOAD_FAILURE);
      expect(result.llmContent).toContain('Error uploading file');
    });
  });
});
