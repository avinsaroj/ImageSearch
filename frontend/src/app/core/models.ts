export type Tri = 'all' | 'yes' | 'no';
export type RankMode = 'fusion' | 'clip' | 'dino';

export interface ProductImage {
  ImgID: number;
  ImageURL: string;
  ImgViewID: number | null;
  score?: number;
}

export interface ProductResult {
  rank: number;
  item_id: number;
  item_code: string;
  category_code: string;
  is_plain_gold: boolean | null;
  is_solitaire: boolean | null;
  is_valid: boolean | null;
  is_franchise_item: boolean | null;
  status_remark: string | null;
  score: number;
  score_per_space: Record<string, number>;
  matched_images: number;
  best_image: ProductImage;
  images: ProductImage[];
}

export interface SearchResponse {
  total_returned: number;
  score_type: string;
  mode: RankMode;
  results: ProductResult[];
  image_filename?: string;
}

export interface CategoryGroup {
  category: string;
  results: ProductResult[];
}

export interface CrossCategoryResponse {
  per_category: number;
  groups: CategoryGroup[];
  image_filename?: string;
}

export interface SimilarResponse {
  item_id: number;
  query_images: number;
  total_returned: number;
  results: ProductResult[];
}

export interface SearchParams {
  top_k: number;
  mode: RankMode;
  category: string;
  plain_gold: Tri;
  solitaire: Tri;
  valid: Tri;
  franchise: Tri;
  item_codes: string;
  min_score: number;
}

export interface Categories {
  count: number;
  categories: { code: string; normalized: string }[];
}

export interface JobCounters {
  products_scanned?: number;
  images_scanned?: number;
  images_removed?: number;
  to_process?: number;
  processed?: number;
  indexed?: number;
  reused?: number;
  skipped?: number;
  failed?: number;
}

export interface SyncScope {
  categories: string[];
  plain_gold: boolean | null;
  solitaire: boolean | null;
  valid: boolean | null;
  franchise: boolean | null;
  search_text: string | null;
  max_products: number | null;
}

/** Scopes already added to the index; the daily/refresh job re-scans all of them. */
export interface ScopeList {
  scopes: Partial<SyncScope>[];
  source: 'saved' | 'env';
}

export interface Job {
  job_id: string;
  kind: 'full' | 'incremental';
  status: string;
  trigger: string;
  stage: string;
  started_at: string;
  finished_at: string | null;
  heartbeat_at: string;
  counters: JobCounters;
  params: SyncScope | null;
  error: string | null;
}

export interface SyncStatus {
  current_job: Job | null;
  last_success: Job | null;
  model_version: string;
  sql_server: { products: number | null; images: number | null; error?: string };
  images: { indexed: number; pending: number; failed: number; removed: number };
  qdrant: { status: string; alias?: string; collection?: string; points?: number; error?: string };
  scheduler: { timezone: string; daily_at: string; last_fired_date: string | null };
}

export interface FailedImage {
  img_id: number;
  item_id: number;
  url: string;
  attempts: number;
  last_error: string | null;
  last_processed_at: string | null;
}

export interface ScopePreview {
  products: number;
  images: number;
  files: number;
}
