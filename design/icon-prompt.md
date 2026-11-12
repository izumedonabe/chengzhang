# 成章 App 图标提示词

## 交给我时的要求

- 1024 × 1024 PNG，**正方形、铺满画面、不要圆角、不要透明背景**（圆角和留边我来做，做成 macOS 的 .icns）
- 画面里**不要任何文字**
- 缩到 32 × 32 时还能一眼认出：一个太阳加下面几道倒影

## 方案 A（推荐）：日出与倒影

**英文提示词**（Midjourney / GPT 图像 / 即梦等都可以用英文）：

```
App icon artwork, full-bleed square, no rounded corners, no text.
Inspired by Claude Monet's "Impression, Sunrise": a soft misty harbor at dawn in muted blue-grey and grey-green,
a single small vermilion-orange sun disc in the upper middle, and directly below it on the water
three to four short, broken horizontal brush strokes of the same orange as its reflection.
Visible loose oil-paint brushwork, thick impasto texture on the strokes, calm and quiet mood,
very simple composition with lots of empty misty space, centered, flat lighting, high contrast between the orange sun and the grey mist,
reads clearly at small sizes. 1024x1024.
```

**负面提示词**（工具支持时填）：

```
text, letters, watermark, signature, frame, border, rounded corners, photorealistic, 3D render, glossy, gradient mesh,
boats, people, buildings, busy details, multiple suns, neon, cartoon outline
```

**中文版**（国产工具用）：

```
App 图标画面，正方形铺满，不要圆角，不要任何文字。灵感来自莫奈《印象·日出》：黎明时分雾气朦胧的港湾，
灰蓝和灰绿的低饱和色调；画面中上方一个小小的朱橙色圆形太阳，正下方水面上有三到四道同样橙色、断断续续的横向笔触作为倒影。
保留松散的油画笔触和厚涂质感，安静克制，构图极简、大量留白，太阳的橙色和雾气的灰色对比清楚，缩小后依然一眼能认出。
```

## 方案 B：一枚印章

如果想和 App 里的落款呼应：

```
App icon artwork, full-bleed square, no rounded corners.
A traditional Chinese seal impression (印章) in vermilion red, slightly uneven ink and paper texture,
stamped on a pale misty blue-grey background with faint horizontal Monet-like brush strokes,
the seal is a simple rounded square with an abstract sun circle carved inside instead of characters,
minimal, centered, reads clearly at small sizes. No text. 1024x1024.
```

（印章里不要让 AI 写字，生成的汉字几乎都会写错，用一个圆代替"太阳"更稳。）

## 第二版：元素放大（第一张图太阳只占宽度 13%，缩小后认不出）

第一张图试裁过：`design/icon/cropped-*.png`。主要的改动是**把比例直接写进提示词**，并且去掉港口和船。

```
App icon artwork, full-bleed square, no rounded corners, no text.
Monet "Impression, Sunrise" style oil painting, extremely simple composition with only two elements:
1) one large vermilion-orange sun disc, its diameter about 30% of the image width, centered horizontally,
   its center at about 32% from the top;
2) directly below it on calm water, three bold horizontal broken brush strokes of the same orange as its reflection,
   each stroke about 30–40% of the image width, getting slightly shorter downward, placed between 62% and 85% from the top.
Background: soft misty blue-grey and grey-green sky and water with a faint horizon line, loose visible brushwork,
thick impasto on the sun and the strokes, strong contrast between orange and grey.
No harbor, no cranes, no boats, no buildings, nothing else. Must read clearly at 32x32 pixels. 1024x1024.
```

负面提示词：

```
text, letters, watermark, signature, frame, border, rounded corners, harbor, crane, boat, ship, buildings, people,
small sun, tiny details, photorealistic, 3D render, glossy, multiple suns
```

## 第三版：毛玻璃风格（日 / 夜两张）

思路：保留"太阳 + 三道倒影"，改成 macOS 27 液态玻璃那种简洁、半透明的画法；夜间版把太阳换成月亮。
**两张的构图必须完全一样**（位置、大小都相同），这样白天和夜间切换时只是换了光。之后再用 Icon Composer 合成。

几何规格（两张都一样）：
- 圆盘直径约占宽度 32%，水平居中，圆心在距顶部 34% 处
- 三道圆角横条（胶囊形），水平居中，宽度依次约 44% / 32% / 20%，厚度约 5%，分布在距顶部 62%–82% 之间
- 圆盘和横条之间留一段空白，表示水面
- 背景：上下两段柔和渐变（天空 / 水面），中间一条很淡的水平线

### 白天（Light）

```
macOS app icon artwork in Apple's Liquid Glass style, full-bleed square, no rounded corners, no border, no text.
Extremely minimal: one glossy translucent vermilion-orange glass disc (the sun), its diameter about 32% of the width,
centered horizontally with its center 34% from the top; below it, three horizontal rounded glass capsules (its reflection
on water) in the same orange, centered, widths about 44%, 32% and 20% of the width, each about 5% tall, placed between
62% and 82% from the top. Background: soft frosted misty blue-grey gradient, slightly lighter sky above and slightly deeper
grey-green water below, separated by a faint horizon line. Frosted glass material with subtle refraction, soft specular
highlight on the top edge of each shape, gentle inner glow, clean vector shapes, no texture noise, calm and quiet.
Inspired by Monet's "Impression, Sunrise" palette. 1024x1024.
```

### 夜间（Dark）

```
macOS app icon artwork in Apple's Liquid Glass style, dark appearance, full-bleed square, no rounded corners, no border, no text.
Exactly the same composition as a day version: one glossy translucent glass disc (a full moon) in warm pale ivory with a faint
cool halo, its diameter about 32% of the width, centered horizontally with its center 34% from the top; below it, three
horizontal rounded glass capsules (the moon's reflection on water) in pale silver-blue, centered, widths about 44%, 32% and 20%,
each about 5% tall, placed between 62% and 82% from the top. Background: deep night blue to indigo frosted gradient, slightly
lighter sky above and darker water below, a faint horizon line, optionally one tiny soft star in the upper left. Frosted glass
material, subtle refraction, soft specular highlights on top edges, clean vector shapes, no texture noise, calm and quiet. 1024x1024.
```

负面提示词（两张通用）：

```
text, letters, watermark, signature, rounded corners, border, frame, oil paint texture, brush strokes, noise, grain,
photorealistic, 3D clay, cartoon outline, boats, buildings, people, multiple suns or moons, busy background, harsh shadows
```

### 中文版（国产工具用）

白天：

```
macOS 应用图标画面，苹果液态玻璃风格，正方形铺满，不要圆角、不要边框、不要文字。极简：一个光亮半透明的朱橙色玻璃圆盘（太阳），直径约占宽度 32%，水平居中，圆心距顶部 34%；下方三道同色的圆角玻璃横条（水面倒影），水平居中，宽度依次约 44%、32%、20%，厚度约 5%，位于距顶部 62% 到 82% 之间。背景是柔和的磨砂雾灰蓝渐变，上方天空稍亮，下方水面稍深偏灰绿，中间一条很淡的水平线。磨砂玻璃质感，轻微折射，每个形状上沿有柔和高光，干净的矢量形状，没有颗粒噪点，安静克制。配色取自莫奈《印象·日出》。
```

夜间：

```
与白天版构图完全相同的 macOS 应用图标，苹果液态玻璃风格，深色外观，正方形铺满，不要圆角、不要边框、不要文字。太阳换成一轮暖象牙白的半透明玻璃满月，带一圈淡淡的冷色光晕，大小和位置与白天版相同；下方三道淡银蓝色的圆角玻璃横条作为月光倒影，尺寸和位置与白天版相同。背景是深夜蓝到靛蓝的磨砂渐变，天空稍亮、水面更暗，一条很淡的水平线，左上角可以有一颗很小很柔和的星。磨砂玻璃质感，轻微折射，形状上沿柔和高光，干净矢量，没有颗粒噪点。
```
