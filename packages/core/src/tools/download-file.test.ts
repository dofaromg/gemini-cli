/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { DownloadFileToolParams } from './download-file.js';
import { DownloadFileTool } from './download-file.js';
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
const { mockDownload, mockFilesApi } = vi.hoisted(() => {
  const mockDownload = vi.fn();
  const mockFilesApi = {
    download: mockDownload,
  };
  return { mockDownload, mockFilesApi };
});

vi.mock('@google/genai', () => ({
  GoogleGenAI: vi.fn().mockImplementation(() => ({
    files: mockFilesApi,
  })),
}));

describe('DownloadFileTool', () => {
  let tempRootDir: string;
  let tool: DownloadFileTool;
  const abortSignal = new AbortController().signal;

  beforeEach(async () => {
    vi.resetAllMocks();
    
    // Create a unique temporary root directory for each test run
    tempRootDir = await fsp.mkdtemp(
      path.join(os.tmpdir(), 'download-file-tool-root-'),
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
    tool = new DownloadFileTool(mockConfigInstance);
  });

  afterEach(async () => {
    // Clean up the temporary root directory
    if (fs.existsSync(tempRootDir)) {
      await fsp.rm(tempRootDir, { recursive: true, force: true });
    }
  });

  describe('build', () => {
    it('should return an invocation for valid params', () => {
      const params: DownloadFileToolParams = {
        file_uri: 'files/test123',
        download_path: path.join(tempRootDir, 'downloaded.txt'),
      };
      const result = tool.build(params);
      expect(typeof result).not.toBe('string');
    });

    it('should throw error if download path is relative', () => {
      const params: DownloadFileToolParams = {
        file_uri: 'files/test123',
        download_path: 'relative/path.txt',
      };
      expect(() => tool.build(params)).toThrow(
        'Download path must be absolute, but was relative: relative/path.txt. You must provide an absolute path.',
      );
    });

    it('should throw error if download path is outside workspace', () => {
      const params: DownloadFileToolParams = {
        file_uri: 'files/test123',
        download_path: '/outside/root.txt',
      };
      expect(() => tool.build(params)).toThrow(
        /Download path must be within one of the workspace directories/,
      );
    });

    it('should throw error if parent directory does not exist', () => {
      const params: DownloadFileToolParams = {
        file_uri: 'files/test123',
        download_path: path.join(tempRootDir, 'nonexistent', 'file.txt'),
      };
      expect(() => tool.build(params)).toThrow(
        /Parent directory does not exist/,
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
      const vertexTool = new DownloadFileTool(mockConfigInstance);
      
      const params: DownloadFileToolParams = {
        file_uri: 'files/test123',
        download_path: path.join(tempRootDir, 'file.txt'),
      };
      expect(() => vertexTool.build(params)).toThrow(
        /File download is not supported when using Vertex AI/,
      );
    });

    it('should allow downloads to project temp directory', async () => {
      const tempDir = path.join(tempRootDir, '.temp');
      await fsp.mkdir(tempDir, { recursive: true });
      
      const params: DownloadFileToolParams = {
        file_uri: 'files/test123',
        download_path: path.join(tempDir, 'downloaded.txt'),
      };
      const result = tool.build(params);
      expect(typeof result).not.toBe('string');
    });

    it('should throw error if file_uri is empty', () => {
      const params: DownloadFileToolParams = {
        file_uri: '',
        download_path: path.join(tempRootDir, 'file.txt'),
      };
      expect(() => tool.build(params)).toThrow(
        "The 'file_uri' parameter must be non-empty.",
      );
    });

    it('should throw error if download_path is empty', () => {
      const params: DownloadFileToolParams = {
        file_uri: 'files/test123',
        download_path: '',
      };
      expect(() => tool.build(params)).toThrow(
        "The 'download_path' parameter must be non-empty.",
      );
    });
  });

  describe('execute', () => {
    it('should download file successfully', async () => {
      mockDownload.mockResolvedValue(undefined);

      const downloadPath = path.join(tempRootDir, 'downloaded.txt');
      const params: DownloadFileToolParams = {
        file_uri: 'files/test123',
        download_path: downloadPath,
      };
      
      const invocation = tool.build(params);
      const result = await invocation.execute(abortSignal);
      
      expect(result.error).toBeUndefined();
      expect(result.llmContent).toContain('Successfully downloaded file');
      expect(result.llmContent).toContain('files/test123');
      expect(mockDownload).toHaveBeenCalledWith({
        file: 'files/test123',
        downloadPath: downloadPath,
        config: {
          abortSignal: abortSignal,
        },
      });
    });

    it('should handle download errors', async () => {
      mockDownload.mockRejectedValue(new Error('Download failed'));

      const downloadPath = path.join(tempRootDir, 'downloaded.txt');
      const params: DownloadFileToolParams = {
        file_uri: 'files/test123',
        download_path: downloadPath,
      };
      
      const invocation = tool.build(params);
      const result = await invocation.execute(abortSignal);
      
      expect(result.error).toBeDefined();
      expect(result.error?.type).toBe(ToolErrorType.FILE_DOWNLOAD_FAILURE);
      expect(result.llmContent).toContain('Error downloading file');
    });
  });
});
