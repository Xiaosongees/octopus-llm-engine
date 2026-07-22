(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var bg = style.getPropertyValue('--bg').trim();

  // --- Chart: Latency ---
  var chartLatency = echarts.init(document.getElementById('chart-latency'), null, { renderer: 'svg' });
  chartLatency.setOption({
    animation: false,
    tooltip: { appendToBody: true, trigger: 'axis', backgroundColor: '#1A1D2E', borderColor: rule, textStyle: { color: ink, fontSize: 13 } },
    grid: { left: 60, right: 30, top: 30, bottom: 50 },
    xAxis: {
      type: 'category',
      data: ['Dense (128→256)', 'ReLU', 'Softmax', 'MatMul (64x64)', 'LayerNorm', 'Conv2D (3x3)'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11, rotate: 20 },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      name: 'ms',
      nameTextStyle: { color: muted, fontSize: 12 },
      axisLine: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: 'RustAI Engine',
        type: 'bar',
        data: [2.1, 0.3, 0.8, 1.5, 0.6, 3.2],
        itemStyle: { color: accent, borderRadius: [4, 4, 0, 0] },
        barWidth: '28%'
      },
      {
        name: 'Python (NumPy)',
        type: 'bar',
        data: [5.8, 1.1, 2.3, 4.2, 1.7, 7.9],
        itemStyle: { color: muted, borderRadius: [4, 4, 0, 0], opacity: 0.5 },
        barWidth: '28%'
      }
    ],
    legend: {
      data: ['RustAI Engine', 'Python (NumPy)'],
      top: 0,
      textStyle: { color: muted, fontSize: 12 },
      itemWidth: 12,
      itemHeight: 12
    }
  });
  window.addEventListener('resize', function() { chartLatency.resize(); });

  // --- Chart: Throughput ---
  var chartThroughput = echarts.init(document.getElementById('chart-throughput'), null, { renderer: 'svg' });
  chartThroughput.setOption({
    animation: false,
    tooltip: { appendToBody: true, trigger: 'axis', backgroundColor: '#1A1D2E', borderColor: rule, textStyle: { color: ink, fontSize: 13 } },
    grid: { left: 60, right: 30, top: 30, bottom: 50 },
    xAxis: {
      type: 'category',
      data: ['1 线程', '2 线程', '4 线程', '8 线程'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 12 },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      name: 'qps',
      nameTextStyle: { color: muted, fontSize: 12 },
      axisLine: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: 'RustAI Engine',
        type: 'line',
        data: [1200, 2100, 3800, 6500],
        smooth: true,
        symbol: 'circle',
        symbolSize: 8,
        lineStyle: { color: accent, width: 3 },
        itemStyle: { color: accent },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: accent + '40' },
              { offset: 1, color: accent + '05' }
            ]
          }
        }
      },
      {
        name: 'Python (NumPy)',
        type: 'line',
        data: [500, 850, 1400, 2100],
        smooth: true,
        symbol: 'circle',
        symbolSize: 8,
        lineStyle: { color: muted, width: 2, type: 'dashed' },
        itemStyle: { color: muted }
      }
    ],
    legend: {
      data: ['RustAI Engine', 'Python (NumPy)'],
      top: 0,
      textStyle: { color: muted, fontSize: 12 },
      itemWidth: 12,
      itemHeight: 12
    }
  });
  window.addEventListener('resize', function() { chartThroughput.resize(); });
})();