/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { ListFilesToolParams } from './list-files.js';
import { ListFilesTool } from './list-files.js';
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
const { mockList, mockFilesApi } = vi.hoisted(() => {
  const mockList = vi.fn();
  const mockFilesApi = {
    list: mockList,
  };
  return { mockList, mockFilesApi };
});

vi.mock('@google/genai', () => ({
  GoogleGenAI: vi.fn().mockImplementation(() => ({
    files: mockFilesApi,
  })),
}));

describe('ListFilesTool', () => {
  let tempRootDir: string;
  let tool: ListFilesTool;
  const abortSignal = new AbortController().signal;

  beforeEach(async () => {
    vi.resetAllMocks();
    
    // Create a unique temporary root directory for each test run
    tempRootDir = await fsp.mkdtemp(
      path.join(os.tmpdir(), 'list-files-tool-root-'),
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
    tool = new ListFilesTool(mockConfigInstance);
  });

  afterEach(async () => {
    // Clean up the temporary root directory
    if (fs.existsSync(tempRootDir)) {
      await fsp.rm(tempRootDir, { recursive: true, force: true });
    }
  });

  describe('build', () => {
    it('should return an invocation for valid params without page_size', () => {
      const params: ListFilesToolParams = {};
      const result = tool.build(params);
      expect(typeof result).not.toBe('string');
    });

    it('should return an invocation for valid params with page_size', () => {
      const params: ListFilesToolParams = {
        page_size: 10,
      };
      const result = tool.build(params);
      expect(typeof result).not.toBe('string');
    });

    it('should throw error if page_size is negative', () => {
      const params: ListFilesToolParams = {
        page_size: -1,
      };
      expect(() => tool.build(params)).toThrow(
        'page_size must be a positive number',
      );
    });

    it('should throw error if page_size is zero', () => {
      const params: ListFilesToolParams = {
        page_size: 0,
      };
      expect(() => tool.build(params)).toThrow(
        'page_size must be a positive number',
      );
    });

    it('should throw error when using Vertex AI', () => {
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
      const vertexTool = new ListFilesTool(mockConfigInstance);
      
      const params: ListFilesToolParams = {};
      expect(() => vertexTool.build(params)).toThrow(
        /File listing is not supported when using Vertex AI/,
      );
    });
  });

  describe('execute', () => {
    it('should list files successfully', async () => {
      const mockFiles = [
        {
          name: 'files/test123',
          uri: 'https://generativelanguage.googleapis.com/v1beta/files/test123',
          displayName: 'test1.txt',
          mimeType: 'text/plain',
          sizeBytes: 100,
          state: 'ACTIVE',
          createTime: '2024-01-01T00:00:00Z',
        },
        {
          name: 'files/test456',
          uri: 'https://generativelanguage.googleapis.com/v1beta/files/test456',
          displayName: 'test2.mp4',
          mimeType: 'video/mp4',
          sizeBytes: 1000000,
          state: 'ACTIVE',
          createTime: '2024-01-02T00:00:00Z',
        },
      ];

      // Create async iterator for the mock
      async function* mockAsyncIterator() {
        for (const file of mockFiles) {
          yield file;
        }
      }

      mockList.mockReturnValue(mockAsyncIterator());

      const params: ListFilesToolParams = {
        page_size: 10,
      };
      
      const invocation = tool.build(params);
      const result = await invocation.execute(abortSignal);
      
      expect(result.error).toBeUndefined();
      expect(result.llmContent).toContain('Found 2 file(s)');
      expect(result.llmContent).toContain('test1.txt');
      expect(result.llmContent).toContain('test2.mp4');
      expect(result.llmContent).toContain('text/plain');
      expect(result.llmContent).toContain('video/mp4');
      expect(mockList).toHaveBeenCalledWith({
        config: {
          pageSize: 10,
          abortSignal: abortSignal,
        },
      });
    });

    it('should handle empty file list', async () => {
      // Create async iterator that yields nothing
      async function* mockEmptyIterator() {
        // Empty
      }

      mockList.mockReturnValue(mockEmptyIterator());

      const params: ListFilesToolParams = {};
      
      const invocation = tool.build(params);
      const result = await invocation.execute(abortSignal);
      
      expect(result.error).toBeUndefined();
      expect(result.llmContent).toContain('No files found');
    });

    it('should handle list errors', async () => {
      mockList.mockRejectedValue(new Error('List failed'));

      const params: ListFilesToolParams = {};
      
      const invocation = tool.build(params);
      const result = await invocation.execute(abortSignal);
      
      expect(result.error).toBeDefined();
      expect(result.error?.type).toBe(ToolErrorType.FILE_LIST_FAILURE);
      expect(result.llmContent).toContain('Error listing files');
    });
  });
});
