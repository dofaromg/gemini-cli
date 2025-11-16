/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'node:path';
import fs from 'node:fs';
import { GoogleGenAI } from '@google/genai';
import type { File as GeminiFile } from '@google/genai';
import { makeRelative, shortenPath } from '../utils/paths.js';
import type { ToolInvocation, ToolResult } from './tools.js';
import { BaseDeclarativeTool, BaseToolInvocation, Kind } from './tools.js';
import type { Config } from '../config/config.js';
import { ToolErrorType } from './tool-error.js';
import { getErrorMessage } from '../utils/errors.js';
import { AuthType } from '../core/contentGenerator.js';

/**
 * Parameters for the UploadFile tool
 */
export interface UploadFileToolParams {
  /**
   * The absolute path to the file to upload
   */
  absolute_path: string;

  /**
   * Optional display name for the uploaded file
   */
  display_name?: string;
}

class UploadFileToolInvocation extends BaseToolInvocation<
  UploadFileToolParams,
  ToolResult
> {
  constructor(
    private config: Config,
    params: UploadFileToolParams,
  ) {
    super(params);
  }

  getDescription(): string {
    const relativePath = makeRelative(
      this.params.absolute_path,
      this.config.getTargetDir(),
    );
    return `Uploading ${shortenPath(relativePath)} to Gemini API`;
  }

  async execute(signal: AbortSignal): Promise<ToolResult> {
    const contentGeneratorConfig = this.config.getContentGeneratorConfig();
    
    // Check if using Vertex AI - file upload is not supported
    if (contentGeneratorConfig.vertexai) {
      const errorMsg = 'File upload is not supported when using Vertex AI. Please use Gemini API (GEMINI_API_KEY) or OAuth authentication.';
      return {
        llmContent: errorMsg,
        returnDisplay: errorMsg,
        error: {
          message: errorMsg,
          type: ToolErrorType.UPLOAD_NOT_SUPPORTED,
        },
      };
    }

    try {
      // Create a GoogleGenAI instance to access the files API
      const googleGenAI = new GoogleGenAI({
        apiKey: contentGeneratorConfig.apiKey === '' ? undefined : contentGeneratorConfig.apiKey,
        vertexai: contentGeneratorConfig.vertexai,
      });

      // Upload the file
      const uploadedFile: GeminiFile = await googleGenAI.files.upload({
        file: this.params.absolute_path,
        config: {
          abortSignal: signal,
          displayName: this.params.display_name,
        },
      });

      const fileName = path.basename(this.params.absolute_path);
      const displayName = uploadedFile.displayName || fileName;
      const fileUri = uploadedFile.uri || uploadedFile.name;

      const successMessage = `Successfully uploaded file: ${displayName}
File URI: ${fileUri}
MIME Type: ${uploadedFile.mimeType}
Size: ${uploadedFile.sizeBytes ? `${uploadedFile.sizeBytes} bytes` : 'unknown'}
State: ${uploadedFile.state}

You can now reference this file in your prompts using the file URI.`;

      return {
        llmContent: successMessage,
        returnDisplay: successMessage,
      };
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      const errorMsg = `Error uploading file '${this.params.absolute_path}': ${errorMessage}`;
      
      return {
        llmContent: errorMsg,
        returnDisplay: errorMsg,
        error: {
          message: errorMsg,
          type: ToolErrorType.FILE_UPLOAD_FAILURE,
        },
      };
    }
  }
}

/**
 * Implementation of the UploadFile tool logic
 */
export class UploadFileTool extends BaseDeclarativeTool<
  UploadFileToolParams,
  ToolResult
> {
  static readonly Name: string = 'upload_file';

  constructor(private config: Config) {
    super(
      UploadFileTool.Name,
      'UploadFile',
      `Uploads a local file to the Gemini API file manager. This allows you to use large files (videos, audio, images, documents) in conversations without embedding them directly. The uploaded file will be stored in the Gemini API and can be referenced by its URI in subsequent prompts. Note: This tool is only available when using Gemini API authentication (not Vertex AI).`,
      Kind.Other,
      {
        properties: {
          absolute_path: {
            description:
              "The absolute path to the file to upload (e.g., '/home/user/project/video.mp4'). Relative paths are not supported.",
            type: 'string',
          },
          display_name: {
            description:
              'Optional: A human-readable display name for the uploaded file. If not provided, the filename will be used.',
            type: 'string',
          },
        },
        required: ['absolute_path'],
        type: 'object',
      },
    );
  }

  protected override validateToolParamValues(
    params: UploadFileToolParams,
  ): string | null {
    const filePath = params.absolute_path;
    
    if (params.absolute_path.trim() === '') {
      return "The 'absolute_path' parameter must be non-empty.";
    }

    if (!path.isAbsolute(filePath)) {
      return `File path must be absolute, but was relative: ${filePath}. You must provide an absolute path.`;
    }

    const workspaceContext = this.config.getWorkspaceContext();
    const projectTempDir = this.config.storage.getProjectTempDir();
    const resolvedFilePath = path.resolve(filePath);
    const resolvedProjectTempDir = path.resolve(projectTempDir);
    const isWithinTempDir =
      resolvedFilePath.startsWith(resolvedProjectTempDir + path.sep) ||
      resolvedFilePath === resolvedProjectTempDir;

    if (!workspaceContext.isPathWithinWorkspace(filePath) && !isWithinTempDir) {
      const directories = workspaceContext.getDirectories();
      return `File path must be within one of the workspace directories: ${directories.join(', ')} or within the project temp directory: ${projectTempDir}`;
    }

    const fileService = this.config.getFileService();
    if (fileService.shouldGeminiIgnoreFile(params.absolute_path)) {
      return `File path '${filePath}' is ignored by .geminiignore pattern(s).`;
    }

    // Check if file exists
    try {
      if (!fs.existsSync(params.absolute_path)) {
        return `File does not exist: ${filePath}`;
      }
      const stats = fs.lstatSync(params.absolute_path);
      if (!stats.isFile()) {
        return `Path is not a file: ${filePath}`;
      }
    } catch (error) {
      return `File does not exist or cannot be accessed: ${filePath}`;
    }

    // Check authentication type
    const contentGeneratorConfig = this.config.getContentGeneratorConfig();
    if (contentGeneratorConfig.vertexai) {
      return 'File upload is not supported when using Vertex AI. Please use Gemini API (GEMINI_API_KEY) or OAuth authentication.';
    }

    if (
      contentGeneratorConfig.authType !== AuthType.USE_GEMINI &&
      contentGeneratorConfig.authType !== AuthType.LOGIN_WITH_GOOGLE
    ) {
      return 'File upload requires either Gemini API key (GEMINI_API_KEY) or OAuth authentication.';
    }

    return null;
  }

  protected createInvocation(
    params: UploadFileToolParams,
  ): ToolInvocation<UploadFileToolParams, ToolResult> {
    return new UploadFileToolInvocation(this.config, params);
  }
}
