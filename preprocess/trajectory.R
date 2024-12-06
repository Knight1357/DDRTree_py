#!/usr/bin/env Rscript

" 拟时序分析
Usage:
  trajectory.R -i <object> -o <dir> --meta_data <file> --root_type <str> [--column <str>] [--method <str>] [-q <int>]
Options:
  -i,--input=file             聚类后的seurat对象
  -o,--outdir=dir             结果输出目录
  --meta_data=file
        meta data数据，第一列为细胞
        且需要包含--column指定列名的信息
  --root_type=str
        根节点细胞的类型
  --column=str
        类型所在列名
        [default: cell_type]
  -q,--qval=int
        筛选差异基因qvalue值
        [default: 0.01]
  --method=str
        拟时方法，可选 monocle3,monocle2
        [default: monocle3]
" -> doc
suppressMessages({
  library(monocle3)
  library(Seurat)
  library(ggplot2)
  library(dplyr)
  library(Matrix)
  library(ComplexHeatmap)
})

# 加载脚本
if (interactive()) {
  file <- whereami::whereami()
} else {
  file <- whereami::thisfile()
}

for (script in list.files(file.path(dirname(normalizePath(file, mustWork = T)), "../R"), full.names = T)) {
  source(script)
}

args <- docopt::docopt(doc)
args <- validate_args(args, type_list = list(
  outdir = "file",
  qval = "number"
))


out_column <- c("orig.ident", "seurat_clusters", args$column, "nCount_RNA", "nFeature_RNA")

select_root_cell <- function(cds, root_type, reduction_method = "UMAP") {
  index <- which(colData(cds)$ident %in% root_type)
  if (length(index) == 0) {
    stop(root_type, " not in meta data.")
  }
  cds <- cds[, index]
  reduction_data <- reducedDim(cds, reduction_method)
  # 获取中心
  ncenter <- colMeans(reduction_data)
  # 获取距离中心点最小的细胞
  index <- which.min(sqrt((reduction_data[,1] - ncenter[1])^2 + (reduction_data[,2] - ncenter[2])^2))
  root_cell <- rownames(reduction_data)[index]
  root_cell
}

plot_pseudotime_heatmap <- function(cds,
                                    features, trend_formula = "~ splines::ns(pseudotime, df=3) + orig.ident", filename = "pseudotime_heatmap.pdf") {
  colData(cds)$pseudotime <- pseudotime(cds)
  cds <- cds[, order(pseudotime(cds))]
  cds_subset <- cds[features, is.finite(colData(cds)$pseudotime)]
  new_data <- as.data.frame(colData(cds_subset))
  new_data$Size_Factor <- 1
  model_tbl <- fit_models(cds_subset,
    model_formula_str = trend_formula
  )

  model_expectation <- model_predictions(model_tbl, new_data = new_data)
  # testthat::expect_equal(t(scale(t(as.matrix(model_expectation)))), ScaleData(model_expectation), check.attributes = FALSE)
  # 标准化
  model_expectation <- Seurat:::ScaleData.default(model_expectation)
  top_anno <- HeatmapAnnotation(
    line = anno_empty(border = F)
  )
  # row_tree = hclust(dist(model_expectation),method = "ward.D2")
  # cluster <- cutree(row_tree,k = 2)
  colors <- rev(RColorBrewer::brewer.pal(11, "Spectral"))

  ht <- ComplexHeatmap::Heatmap(
    model_expectation,
    name = "z-score",
    col = circlize::colorRamp2(seq(from = -2, to = 2, length = 11), colors),
    show_row_names = TRUE,
    show_column_names = FALSE,
    row_names_gp = gpar(fontsize = 6),
    clustering_method_rows = "ward.D2",
    row_title_rot = 0,
    cluster_rows = TRUE,
    cluster_row_slices = FALSE,
    cluster_columns = FALSE, 
    top_annotation = top_anno,
    raster_resize_mat = FALSE
  )

  ragg::agg_png(filename, width = 10, height = 8,units = "in",res = 300)
  draw(ht)
  # 增加注释修饰
  # to do ，可以直接使用pushViewport进行拼图
  blues <- colorRampPalette(RColorBrewer::brewer.pal(n = 9, "Blues"))
  draw_lines <- function(x, y, colors) {
    arrow <- NULL
    len <- length(x)
    grobs <- vector(mode = "list", length = len - 1)
    for (i in 1:len) {
      if (i == len) break
      if (i == (len - 1)) arrow <- arrow(length = unit(0.2, "npc"))
      grid.draw(grid.lines(x = c(x[i], x[i + 1]), y = c(y[i], y[i + 1]), arrow = arrow, gp = gpar(lwd = 3, col = colors[i])))
    }
    return(NULL)
  }

  print(decorate_annotation("line", {
    draw_lines(x = seq(0, 0.1, length.out = 11), y = rep(0.2, 11), col = blues(11))
    grid.text("Pseudotime", x = unit(0, "npc"), y = unit(0.8, "npc"), just = "left", gp = gpar(cex = 1.2, fontface = "bold"))
  }))
  dev.off()
  
  invisible(gc())
  # pdf
  pdf(paste0(tools::file_path_sans_ext(filename),".pdf"), width = 10, height = 8)
  draw(ht)
  # 增加注释修饰
  # to do ，可以直接使用pushViewport进行拼图
  blues <- colorRampPalette(RColorBrewer::brewer.pal(n = 9, "Blues"))
  draw_lines <- function(x, y, colors) {
    arrow <- NULL
    len <- length(x)
    grobs <- vector(mode = "list", length = len - 1)
    for (i in 1:len) {
      if (i == len) break
      if (i == (len - 1)) arrow <- arrow(length = unit(0.2, "npc"))
      grid.draw(grid.lines(x = c(x[i], x[i + 1]), y = c(y[i], y[i + 1]), arrow = arrow, gp = gpar(lwd = 3, col = colors[i])))
    }
    return(NULL)
  }
  
  print(decorate_annotation("line", {
    draw_lines(x = seq(0, 0.1, length.out = 11), y = rep(0.2, 11), col = blues(11))
    grid.text("Pseudotime", x = unit(0, "npc"), y = unit(0.8, "npc"), just = "left", gp = gpar(cex = 1.2, fontface = "bold"))
  }))
  dev.off()
}


module_monocle3 <- function(args) {
  object <- process_input(args$input, args$meta_data, args$column)
  # 对于整合的数据，counts的assay为空矩阵
  if (nrow(GetAssayData(object, slot = "counts")) == 0) {
    DefaultAssay(object) <- "RNA"
  }

  # 直接保留counts等三个矩阵以及相关的降维数据
  # scaledata等直接清除？减少内存，后续使用再重新计算
  cds <- suppressWarnings(SeuratWrappers::as.cell_data_set(object))
  # 列名为数字开头时
  if (!(args$column %in% colnames(cds@colData))) {
    column <- make.names(args$column)
    index_col <- which(colnames(cds@colData) == column)
    colnames(cds@colData)[index_col] <- args$column
  }
  cds@assays@data$scaledata <- NULL
  # bug，https://github.com/cole-trapnell-lab/monocle3/issues/602
  cds <- estimate_size_factors(cds)

  cds@reduce_dim_aux$PCA$model$svd_v <- object@reductions$pca@feature.loadings
  cds@reduce_dim_aux$PCA$model$svd_sdev <- object@reductions$pca@stdev
  rm(object);invisible(gc())

  cds <- cluster_cells(cds,
    reduction_method = "UMAP",
    cluster_method = "leiden",
    resolution = NULL
  )

  # 使用PAGA，use_partition参数
  cds <- suppressWarnings(learn_graph(cds, use_partition = T, close_loop = T, verbose = FALSE))

  # 计算伪时间pseudotime
  # 跟节点为选择类的中心
  root_cells <- select_root_cell(cds, args$root_type)
  cds <- order_cells(cds, root_cells = root_cells, verbose = F)

  p <- plot_cells(cds,
    color_cells_by = args$column,
    label_cell_groups = FALSE,
    label_leaves = TRUE,
    label_branch_points = TRUE,
    graph_label_size = 1.5,
    show_trajectory_graph = TRUE
  )
  # 获取前端绘图数据,这样不是很好。。
  # 直接`plot_cells`源码获取数据??
  pdata <- ggplot2::ggplot_build(p)

  # 增加 pseudotime差异基因分析
  modulated_genes <- graph_test(cds, neighbor_graph = "principal_graph", cores = 4)
  modulated_genes <- subset(modulated_genes, q_value < args$qval) |>
    dplyr::arrange(q_value)

  # 绘制热图
  # 选择50个基因太多，改为10个
  genes <- modulated_genes |>
    dplyr::slice_max(n = 10, morans_I) |>
    rownames()

  # 需要使用这种module么
  # 不知道后三列的意思是什么
  subset_cds <- cds[rownames(modulated_genes), ] #不使用管道符传参避免惰性评估引起命名空间冲突
  gene_module_df <- find_gene_modules(subset_cds, resolution = 0.001)

  gene_module_df$module <- paste0("Module ", gene_module_df$module)

  modulated_genes$gene <- rownames(modulated_genes)
  modulated_genes <- modulated_genes |>
    dplyr::left_join(gene_module_df[, c("id", "module")], by = c("gene" = "id")) |>
    dplyr::select(-status) |>
    dplyr::relocate(gene)

  cell_group_df <- data.frame(
    cell = row.names(colData(cds)),
    cell_group = colData(cds)[[args$column]]
  )
  agg_mat <- aggregate_gene_expression(cds, gene_module_df, cell_group_df)
  # module_heatmap <- ComplexHeatmap::pheatmap(agg_mat, scale = "column", clustering_method = "ward.D2")

  in_dir(args$outdir, {
    # 制作示例
    # ggsave("trajectory_cluster.png",plot = p,width = 10,height = 8)
    # p_pseudotime <- plot_cells(cds,
    #   color_cells_by = "pseudotime",
    #   label_cell_groups = FALSE,
    #   label_leaves = TRUE,
    #   label_branch_points = TRUE,
    #   graph_label_size = 1.5,
    #   show_trajectory_graph = T)
    # ggsave("trajectory_pseudotime.png",plot = p_pseudotime,width = 10,height = 8)
    fwrite2(pdata$data[[3]] |>
              dplyr::select(x, xend, y, yend), file = "edges.tsv", keep.rownames = F)
    #--------------
    # 可能会出现 branch points为空的情况
    #--------------
    branch_points <- pdata$data[[7]]
    if (nrow(branch_points) == 0) {
      branch_points <- data.frame(
        x = numeric(),
        y = numeric(),
        label = character()
      )
    } else {
      branch_points <- branch_points[, c("x", "y", "label")]
    }
    fwrite2(branch_points, file = "points.tsv", keep.rownames = F)
    
    cbind(
      data.frame(pseudotime = pseudotime(cds)),
      as.data.frame(colData(cds)[, out_column]),
      as.data.frame(reducedDim(cds, "UMAP"))
    ) |>
      fwrite2("meta_data.tsv", keep.rownames = "cell")

    modulated_genes |>
      mutate(across(c(p_value, q_value),
        .fns = format_scientific
      )) |>
      fwrite2(
        file = "module_diff_gene.tsv",
        keep.rownames = F
      )
    # pdf("module_heatmap.pdf", width = 10, height = 8)
    # print(module_heatmap)
    # dev.off()
    fwrite2(agg_mat,"module_matrix.tsv")
    # 拟时热图
    # 不输出pdf 是由于 12 万细胞导致pdf 无法转换为 png 报错：https://github.com/jokergoo/ComplexHeatmap/issues/952
    # Error in x$.self$finalize() : attempt to apply non-function
    returnmessage <- tryCatch(
    {
      # 运行任务
      plot_pseudotime_heatmap(cds, features = genes, filename = "pseudotime_heatmap.png")
      "normal"  # 返回normal表示正常运行
    },
    error = function(e) {
      # 错误处理代码
      errorMessage <- conditionMessage(e)
      print(errorMessage)
      errorMessage  # 返回错误消息
    }
    )
  
    if(returnmessage=="normal"){
      cat("绘图完成")
    } else{
      if (grepl("attempt to apply non-function", returnmessage)){
        plot_pseudotime_heatmap(cds, features = genes, filename = "pseudotime_heatmap.png")
      } else{
        cat("绘图失败")
            }
    }

    # 输出数据用于绘图
    cds@assays@data$logcounts <- NULL
    qs::qsave(cds, "cds.qs")
    
  })
  invisible()
}

#' seurat转化为monocle2对象
#'
as_monocle2 <- function(object) {
  # 对于整合的数据，counts的assay为空矩阵
  if (nrow(GetAssayData(object, slot = "counts")) == 0) {
    DefaultAssay(object) <- "RNA"
  }

  fData <- data.frame(
    gene_short_name = row.names(object),
    row.names = row.names(object)
  )

  cds <- monocle::newCellDataSet(
    cellData = GetAssayData(object, slot = "counts"),
    phenoData = new("AnnotatedDataFrame", data = object@meta.data),
    featureData = new("AnnotatedDataFrame", data = fData),
    expressionFamily = negbinomial.size()
  )
  cds <- estimateSizeFactors(cds)
  cds <- suppressWarnings(estimateDispersions(cds))
  disp_table <- dispersionTable(cds)
  ordering_genes <- subset(disp_table, mean_expression >= 0.1)
  # 增加高可变基因用于降维
  cds <- setOrderingFilter(cds,
    ordering_genes = ordering_genes
  )
  cds
}

module_monocle2 <- function(args) {
  suppressMessages(require(monocle))

  object <- process_input(args$input, args$meta_data, args$column)
  cds <- as_monocle2(object)
  rm(object);invisible(gc())
  # 降维
  cds <- monocle::reduceDimension(cds,
    max_components = 2,
    method = "DDRTree"
  )
  # 修复monocle:::project2MST在R >= 4.2的bug
  # 修改源代码
  project2MST <- monocle:::project2MST
  body(project2MST)[[11]][[4]][[3]][[4]][[7]] <- substitute(if (!inherits(projection, "matrix")) projection <- as.matrix(projection))
  assignInNamespace("project2MST", project2MST, ns = "monocle")

  # 自己定义的root_state的效果似乎跟NULL没有区别？
  # Biobase::pData(cds)$State <- Biobase::pData(cds)[[args$column]]

  cds <- suppressWarnings(orderCells(cds, root_state = NULL))
  # 绘制拟时序图
  p <- plot_cell_trajectory(cds)
  pdata <- ggplot_build(p)
  # 差异基因分析
  diff_gene <- differentialGeneTest(cds,
                                    fullModelFormulaStr = "~sm.ns(Pseudotime)",
                                    cores = 4
  )
  diff_gene <- diff_gene[, c("pval", "qval")]
  names(diff_gene) <- c("pvalue", "qvalue")
  diff_gene <- subset(diff_gene, qvalue < args$qval) |>
    dplyr::arrange(qvalue)
  
  # 修复拟时热图 在R >= 4.2 bug
  isSparseMatrix <- monocle:::isSparseMatrix
  body(isSparseMatrix)[[2]] <- substitute(is(x, "dgCMatrix"))
  assignInNamespace("isSparseMatrix", isSparseMatrix, ns = "monocle")

  in_dir(args$outdir, {
    # meta_data
    meta_data <- Biobase::pData(cds)
    meta_data <- meta_data[, c("Pseudotime", "State", out_column)]
    names(meta_data)[1] <- "pseudotime"
    Component <- as.data.frame(t(monocle::reducedDimS(cds)))
    names(Component) <- c("Component 1", "Component 2")

    cbind(
      meta_data,
      Component
    ) |>
      fwrite2("meta_data.tsv", keep.rownames = "cell")

    # 拟时点
    pdata$data[[4]] |>
      dplyr::select(x, y, label) |>
      fwrite2(file = "points.tsv", keep.rownames = F)
    # 拟时线段
    pdata$data[[1]] |>
      dplyr::select(x, xend, y, yend) |>
      fwrite2(file = "edges.tsv", keep.rownames = F)
  })


  in_dir(args$outdir, {
    diff_gene |>
      mutate(across(c(pvalue, qvalue),
        .fns = format_scientific
      )) |>
      fwrite2(
        file = "diff_gene.tsv",
        keep.rownames = "gene"
      )

    pdf("pseudotime_heatmap.pdf", width = 10, height = 8)
    monocle::plot_pseudotime_heatmap(
      cds[head(rownames(diff_gene), 50), ],
      num_clusters = 4,
      cores = 4,
      show_rownames = T
    ) |> print()
    dev.off()

    list.files("./", pattern = ".pdf$") |>
      convert_image(to_format = "png")
    qs::qsave(cds, "cds.qs")
  })
}

tryCatch({
  run_time <- bench::bench_time({
    switch(args$method,
      "monocle3" = module_monocle3(args),
      "monocle2" = module_monocle2(args),
      stop("unsport method: ", args$method)
    )
  })
}, error = function(e) {
  message("An error occurred: ", e$message)
  traceback()  # 打印堆栈信息
})

message_time("Trajectory analysis", run_time)
