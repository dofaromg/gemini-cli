/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import type { File as GeminiFile } from '@google/genai';
import type { ToolInvocation, ToolResult } from './tools.js';
import { BaseDeclarativeTool, BaseToolInvocation, Kind } from './tools.js';
import type { Config } from '../config/config.js';
import { ToolErrorType } from './tool-error.js';
import { getErrorMessage } from '../utils/errors.js';
import { AuthType } from '../core/contentGenerator.js';

/**
 * Parameters for the ListFiles tool
 */
export interface ListFilesToolParams {
  /**
   * Optional maximum number of files to return
   */
  page_size?: number;
}

class ListFilesToolInvocation extends BaseToolInvocation<
  ListFilesToolParams,
  ToolResult
> {
  constructor(
    private config: Config,
    params: ListFilesToolParams,
  ) {
    super(params);
  }

  getDescription(): string {
    return 'Listing uploaded files from Gemini API';
  }

  async execute(signal: AbortSignal): Promise<ToolResult> {
    const contentGeneratorConfig = this.config.getContentGeneratorConfig();
    
    // Check if using Vertex AI - file listing is not supported
    if (contentGeneratorConfig.vertexai) {
      const errorMsg = 'File listing is not supported when using Vertex AI. Please use Gemini API (GEMINI_API_KEY) or OAuth authentication.';
      return {
        llmContent: errorMsg,
        returnDisplay: errorMsg,
        error: {
          message: errorMsg,
          type: ToolErrorType.DOWNLOAD_NOT_SUPPORTED,
        },
      };
    }

    try {
      // Create a GoogleGenAI instance to access the files API
      const googleGenAI = new GoogleGenAI({
        apiKey: contentGeneratorConfig.apiKey === '' ? undefined : contentGeneratorConfig.apiKey,
        vertexai: contentGeneratorConfig.vertexai,
      });

      // List the files
      const listResponse = await googleGenAI.files.list({
        config: {
          pageSize: this.params.page_size,
          abortSignal: signal,
        },
      });

      const files: GeminiFile[] = [];
      for await (const file of listResponse) {
        files.push(file);
      }

      if (files.length === 0) {
        const message = 'No files found in the Gemini API file manager.';
        return {
          llmContent: message,
          returnDisplay: message,
        };
      }

      // Format the files list
      const filesList = files
        .map((file, index) => {
          const displayName = file.displayName || 'N/A';
          const uri = file.uri || file.name || 'N/A';
          const mimeType = file.mimeType || 'unknown';
          const size = file.sizeBytes ? `${file.sizeBytes} bytes` : 'unknown';
          const state = file.state || 'unknown';
          const createTime = file.createTime || 'unknown';

          return `${index + 1}. ${displayName}
   URI: ${uri}
   MIME Type: ${mimeType}
   Size: ${size}
   State: ${state}
   Created: ${createTime}`;
        })
        .join('\n\n');

      const successMessage = `Found ${files.length} file(s) in Gemini API file manager:

${filesList}

You can reference these files in your prompts using their URI, or download them using the download_file tool.`;

      return {
        llmContent: successMessage,
        returnDisplay: successMessage,
      };
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      const errorMsg = `Error listing files from Gemini API: ${errorMessage}`;
      
      return {
        llmContent: errorMsg,
        returnDisplay: errorMsg,
        error: {
          message: errorMsg,
          type: ToolErrorType.FILE_LIST_FAILURE,
        },
      };
    }
  }
}

/**
 * Implementation of the ListFiles tool logic
 */
export class ListFilesTool extends BaseDeclarativeTool<
  ListFilesToolParams,
  ToolResult
> {
  static readonly Name: string = 'list_files';

  constructor(private config: Config) {
    super(
      ListFilesTool.Name,
      'ListFiles',
      `Lists all files currently stored in the Gemini API file manager. This shows files that have been previously uploaded and are available for use in conversations. Each file entry includes its URI, display name, MIME type, size, and upload date. Note: This tool is only available when using Gemini API authentication (not Vertex AI).`,
      Kind.Other,
      {
        properties: {
          page_size: {
            description:
              'Optional: Maximum number of files to return. If not specified, all files will be listed.',
            type: 'number',
          },
        },
        required: [],
        type: 'object',
      },
    );
  }

  protected override validateToolParamValues(
    params: ListFilesToolParams,
  ): string | null {
    if (params.page_size !== undefined && params.page_size <= 0) {
      return 'page_size must be a positive number';
    }

    // Check authentication type
    const contentGeneratorConfig = this.config.getContentGeneratorConfig();
    if (contentGeneratorConfig.vertexai) {
      return 'File listing is not supported when using Vertex AI. Please use Gemini API (GEMINI_API_KEY) or OAuth authentication.';
    }

    if (
      contentGeneratorConfig.authType !== AuthType.USE_GEMINI &&
      contentGeneratorConfig.authType !== AuthType.LOGIN_WITH_GOOGLE
    ) {
      return 'File listing requires either Gemini API key (GEMINI_API_KEY) or OAuth authentication.';
    }

    return null;
  }

  protected createInvocation(
    params: ListFilesToolParams,
  ): ToolInvocation<ListFilesToolParams, ToolResult> {
    return new ListFilesToolInvocation(this.config, params);
  }
}
