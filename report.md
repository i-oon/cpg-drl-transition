 

Transition-Aware Quadruped Locomotion:
A Study of Residual Correction Spaces









นายดิษย์ธร  สุทธาเวศ  66340500019










โครงงานนี้เป็นส่วนหนึ่งของการศึกษาตามหลักสูตร
ปริญญาวิศวกรรมศาสตรบัณฑิต  สาขาวิชาวิศวกรรมหุ่นยนต์และระบบอัตโนมัติ
สถาบันวิทยาการหุ่นยนต์ภาคสนาม
มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี
ปีการศึกษา 2568

 
สารบัญ
บทที่ 1 บทนำ	4
1.1 ที่มา ความสำคัญ	4
1.2 ประโยคปัญหางานวิจัย (Problem Statement)	6
1.3 ผลผลิตและผลลัพธ์ (Outputs and Outcomes)	7
ผลผลิต	7
ผลลัพธ์	8
1.4 ความต้องการของระบบ (Requirements)	9
1.5 ขอบเขตของงานวิจัย (Scopes)	9
1.6 ข้อกำหนดของงานวิจัย (Assumptions)	9
1.7 ขั้นตอนการดำเนินงาน	10
บทที่ 2 ทฤษฎี/งานวิจัย/การศึกษาที่เกี่ยวข้อง	11
2.1 Reinforcement Learning และ Proximal Policy Optimization	11
2.1.1 Proximal Policy Optimization (PPO)	11
2.2 Residual Policy Learning	11
2.3 Gait Coordination ของหุ่นยนต์สี่ขา	12
2.4	Smoothstep Interpolation	12
2.5 สถาปัตยกรรมหุ่นยนต์ Unitree B1	13
2.6 งานวิจัยที่เกี่ยวข้อง	14
บทที่ 3 ระเบียบวิธีวิจัย	19
3.1	แนวทางเริ่มต้นและเหตุผลในการปรับเปลี่ยน	19
3.2 แนวทางที่เลือกใช้ (Per-leg Residual Learning)	19
3.3 Phase 1 — Base Gait Policies	20
3.3.1 Custom Reward Engineering สำหรับ B1	20
3.4 Phase 2 — Per-Leg Residual Transition Learning	20
3.5 Per-Leg Blending และ Time-Gating	20
3.6 Reward Function สำหรับ Phase 2	21
3.7 วิธีการประเมินผล	21
บทที่ 4 การทดลองและผลการทดลอง/วิจัย	23
4.1 ผลการเปรียบเทียบ 7 วิธี (Seed=42)	23
4.2 Seed Robustness (60 Windows)	23
4.3 Ablation Study	24
4.4 Duration Sweep	25
4.5 ข้อค้นพบสำคัญ	25
บทที่ 5 บทสรุป	26
5.1 สรุปผลการวิจัย	26
5.1.1 ข้อค้นพบเชิงสถาปัตยกรรม	26
5.2 ข้อจำกัดและแนวทางในอนาคต	27
เอกสารอ้างอิง	28

 
บทที่ 1 บทนำ
1.1 ที่มา ความสำคัญ 
หุ่นยนต์ขา (Legged Robot) ได้รับแรงบันดาลใจจากระบบเคลื่อนที่ทางชีวภาพ (Biological Locomotion) ของสัตว์ ซึ่งสามารถเคลื่อนที่ได้อย่างคล่องแคล่วบนภูมิประเทศหลากหลาย ข้อได้เปรียบหลักของหุ่นยนต์ขาเมื่อเทียบกับหุ่นยนต์ล้อหรือสายพาน คือความสามารถในการเคลื่อนที่บนพื้นผิวที่ไม่สม่ำเสมอ โดยใช้จุดสัมผัสแบบ Discrete Contact กับพื้น ทำให้สามารถเลือกตำแหน่งวางเท้าที่เหมาะสมได้ หุ่นยนต์สี่ขา (Quadruped Robot) โดยเฉพาะ ได้รับความนิยมสูงเนื่องจากมีความเสถียรเชิงสถิตดีกว่าหุ่นยนต์สองขา และมีความคล่องตัวเพียงพอสำหรับงานหลากหลาย ตั้งแต่การสำรวจและตรวจสอบ, การขนส่งสิ่งของ ไปจนถึงภารกิจค้นหาและกู้ภัย ตัวอย่างเช่น Unitree B1 ซึ่งเป็นหุ่นยนต์สี่ขาขนาดกลางน้ำหนัก ~50 กก. ที่ออกแบบมาสำหรับงานอุตสาหกรรมและการวิจัย
การสร้างและเปลี่ยนรูปแบบการเดิน (Gait Generation and Transition) ของหุ่นยนต์ขาได้รับแนวทางหลักจาก 2 กระบวนทัศน์ทางชีวภาพ (Biological Paradigm) ได้แก่ (1) Sensory-Driven Paradigm เป็นการเปลี่ยนรูปแบบการเดินจากการถูกกระตุ้นจากข้อมูลประสาทสัมผัส (Sensory Feedback) เช่น ความเร็วที่ต้องการ สภาพพื้นผิว หรือสัญญาณคำสั่ง ระบบประมวลผลข้อมูลเหล่านี้แล้วเลือกรูปแบบการเดินที่เหมาะสม ซึ่งใน Deep Reinforcement Learning จะแสดงออกเป็นการให้ Policy เรียนรู้ที่จะเลือกและเปลี่ยน Gait จาก Observation โดยตรง และ (2) Coupling-Driven Paradigm เป็นรูปแบบการเดินเกิดจากการประสานจังหวะ (Phase Coupling) ระหว่างขาแต่ละข้าง โดยใช้ Central Pattern Generator (CPG) ที่สร้างสัญญาณ Oscillation แบบคู่ขนาน การเปลี่ยนรูปแบบการเดินทำได้โดยปรับ Phase Offset ระหว่าง Oscillator ของแต่ละขา ซึ่งในงานวิจัยมักใช้ร่วมกับ Evolutionary Optimization เช่น PI^BB (Black-Box Policy Improvement) โครงงานนี้เริ่มต้นด้วยแนวทาง Coupling-Driven (CPG-RBF + PIBB) แต่พบข้อจำกัดเชิงโครงสร้างบนหุ่นยนต์หนัก จึง Pivot มาใช้แนวทาง Sensory-Driven ผ่าน PPO Velocity-Tracking Policy เป็น Base Gait และเสริมด้วย Residual MLP สำหรับการเปลี่ยนรูปแบบการเดิน
ในการใช้งานจริง หุ่นยนต์สี่ขาไม่ควรใช้รูปแบบการเดินเพียงแบบเดียวตลอดเวลา เนื่องจากแต่ละ gait เหมาะกับสถานการณ์ต่างกัน เช่น Trot เหมาะกับการเคลื่อนที่ต่อเนื่องและมีความเสถียร, Bound เหมาะกับการเคลื่อนที่ที่ต้องการแรงผลักไปข้างหน้ามากขึ้น และ Pace เป็นรูปแบบที่ใช้การประสานงานของขาด้านเดียวกัน รูปแบบเหล่านี้ไม่ได้ต่างกันเพียงที่ความเร็ว แต่ต่างกันที่โครงสร้างการประสานงานของขา เช่น Trot ใช้คู่ทแยง FL+RR และ FR+RL, Bound ใช้คู่หน้าและคู่หลัง FL+FR และ RL+RR, ส่วน Pace ใช้คู่ด้านข้าง FL+RL และ FR+RR
 
รูปที่ 1 

ดังนั้นปัญหาสำคัญจึงไม่ใช่เพียงการทำให้หุ่นยนต์เดินได้หลาย gait แต่คือการเปลี่ยนจาก gait หนึ่งไปยังอีก gait หนึ่งอย่างราบรื่น การเปลี่ยน gait แบบทันทีทำให้คำสั่งข้อต่อเปลี่ยนแบบกระโดด ส่งผลให้เกิด Kinematic Shock และ Joint Jerk สูง ในขณะที่การผสมคำสั่งข้อต่อแบบเชิงเส้นก็ยังมีข้อจำกัด เพราะจุดกึ่งกลางระหว่าง gait สองแบบที่มีโครงสร้างการประสานงานต่างกันอาจไม่ใช่ท่าทางที่ถูกต้องของหุ่นยนต์ ตัวอย่างเช่น จุดกึ่งกลางระหว่าง Trot ที่ FL+RR เป็นคู่หลัก กับ Bound ที่ FL+FR เป็นคู่หลัก ไม่ได้บอกอย่างชัดเจนว่าขาแต่ละข้างควรเปลี่ยนคู่ประสานงานเมื่อใด
โครงงานนี้จึงมองปัญหา gait transition เป็นปัญหาเชิงเวลาและการประสานงานของขา มากกว่าปัญหาการหามุมข้อต่อใหม่โดยตรง วิธีที่เสนอคือ Per-Leg Residual Learning โดยใช้ Smoothstep เป็น baseline transition schedule และให้ MLP ขนาดเล็กเรียนรู้ค่าแก้ไข Δα แยกแต่ละขา ค่า α ทำหน้าที่กำหนดสัดส่วนระหว่าง policy ต้นทางและ policy ปลายทาง ดังนั้นการเรียนรู้ Δα จึงเป็นการเรียนรู้จังหวะการเปลี่ยน gait ของแต่ละขา แทนที่จะสร้าง joint command ใหม่ทั้งหมด
งานนี้ทดสอบบนหุ่นยนต์ Unitree B1 ซึ่งเป็นหุ่นยนต์สี่ขาขนาดกลางน้ำหนักประมาณ 50 กิโลกรัม มี 12 องศาอิสระ และมีความไม่สมมาตรระหว่างขาหน้าและขาหลังจากค่า Default Thigh Angle ที่ต่างกัน 0.2 rad ความท้าทายนี้ทำให้การเปลี่ยน gait มีความรุนแรงกว่าหุ่นยนต์ขนาดเล็ก และเป็นเหตุผลที่ต้องใช้สถาปัตยกรรมแบบ Per-leg เพื่อให้แต่ละขาสามารถเปลี่ยนผ่านด้วยอัตราที่ต่างกันได้
โครงงานนี้จึงมีเป้าหมายเพื่อศึกษาว่า Residual Network ขนาดเล็กที่ทำงานใน α-space สามารถลดความรุนแรงของ Gait Transition ได้ดีกว่าวิธีพื้นฐาน เช่น Discrete Switch, Linear Ramp, Smoothstep Ramp และ Residual ที่แก้คำสั่งข้อต่อโดยตรงหรือไม่ โดยใช้ตัวชี้วัดหลักคือTransition-window Joint Jerk, Velocity Reversal, Body Stability และ Cost of Transport

  
รูปที่ 2

 
1.2 ประโยคปัญหางานวิจัย (Problem Statement)
หุ่นยนต์สี่ขาที่ใช้งานจริงต้องสามารถเปลี่ยนรูปแบบการเดินได้ตามคำสั่งหรือสภาพแวดล้อม เช่น จาก Trot ไป Bound หรือจาก Bound ไป Pace โดยไม่สูญเสียสมดุล ไม่เกิดแรงกระชากสูง และไม่เกิดการถอยหลังชั่วขณะระหว่างการเปลี่ยนรูปแบบการเดิน อย่างไรก็ตาม ปัญหานี้มีความยากโดยพื้นฐาน เนื่องจากแต่ละ Gait ไม่ได้ต่างกันเพียงค่ามุมข้อต่อ แต่ต่างกันที่โครงสร้างการประสานงานของขา เช่น Trot ใช้คู่ทแยง FL+RR และ FR+RL, Bound ใช้คู่หน้า-หลัง FL+FR และ RL+RR, ส่วน Pace ใช้คู่ด้านข้าง FL+RL และ FR+RR
ดังนั้น การเปลี่ยน Gait แบบทันที หรือ Discrete Switch ทำให้คำสั่งข้อต่อเปลี่ยนอย่างไม่ต่อเนื่อง ส่งผลให้เกิด Kinematic Shock และ Joint Jerk สูง ส่วนการ Blend ที่ระดับ Joint Target ด้วยค่า α เดียว เช่น Linear Ramp หรือ Smoothstep Ramp แม้จะลดความรุนแรงของการเปลี่ยนคำสั่งได้บางส่วน แต่ยังไม่สามารถแก้ปัญหา Coordination-Partner Swap ได้อย่างเหมาะสม เพราะแต่ละขาอาจต้องเปลี่ยนจาก Gait เดิมไปยัง Gait ใหม่ด้วยอัตราที่แตกต่างกัน
ปัญหานี้ชัดเจนขึ้นบนหุ่นยนต์ Unitree B1 ซึ่งเป็นหุ่นยนต์สี่ขาขนาดประมาณ 50 กิโลกรัม มี 12 องศาอิสระ และมีความไม่สมมาตรเชิงโครงสร้างระหว่างขาหน้าและขาหลังจากค่า Default Thigh Angle ที่ต่างกัน 0.2 rad ความไม่สมมาตรนี้ทำให้ขาแต่ละคู่ตอบสนองต่อการเปลี่ยน Gait ไม่เท่ากัน และทำให้การใช้ค่า Blending เดียวทั้งตัวอาจไม่เพียงพอสำหรับ Transition ที่ราบรื่น
Residual Policy Learning เป็นแนวทางที่สามารถเรียนรู้ค่าแก้ไขบน Baseline Controller ได้ โดยไม่ต้องเรียนรู้ Policy ทั้งหมดใหม่ตั้งแต่ต้น อย่างไรก็ตาม ยังมีคำถามสำคัญว่า Residual ควรทำงานใน Space ใด ระหว่างการแก้คำสั่งข้อต่อโดยตรง หรือ Residual-q กับการแก้จังหวะของการ Blend ระหว่าง Policy ต้นทางและ Policy ปลายทาง หรือ Residual-α ในบริบทของ Gait Transition ปัญหาหลักไม่ได้อยู่ที่การสร้าง Joint Command ใหม่ทั้งหมด แต่อยู่ที่การกำหนดว่าแต่ละขาควรเปลี่ยนจังหวะจาก Gait เดิมไปยัง Gait ใหม่เมื่อใด ดังนั้น Residual-α จึงอาจเหมาะสมกว่า Residual-q เพราะแก้ปัญหาในระดับ Transition Timing โดยตรง

 
รูปที่ 3

โครงงานนี้จึงตั้งคำถามวิจัยหลักว่า 
	เครือข่าย Per-Leg Residual ขนาดเล็กที่เรียนรู้ค่าแก้ไขใน α-space สามารถลดความรุนแรงของการเปลี่ยนรูปแบบการเดินบนหุ่นยนต์ Unitree B1 ได้ดีกว่าวิธีพื้นฐาน เช่น Discrete Switch, Smoothstep Ramp และ Residual ที่แก้ใน Joint Space หรือไม่? 
	เครือข่าย Per-Leg Residual ขนาดเล็กที่เรียนรู้ค่าแก้ไขใน α-space สามารถลดความรุนแรงของการเปลี่ยนรูปแบบการเดินบนหุ่นยนต์ Unitree B1 ได้ดีกว่าวิธีพื้นฐาน เช่น Discrete Switch, Smoothstep Ramp และ Residual ที่แก้ใน Joint Space หรือไม่?

1.3 ผลผลิตและผลลัพธ์ (Outputs and Outcomes)
ผลผลิต
	Base Gait Policy จำนวน 3 รูปแบบ (Trot, Bound, Pace) ที่ฝึกด้วย PPO Velocity-Tracking สำหรับ Unitree B1 บน Isaac Lab

 
รูปที่ 4

 
รูปที่ 5


 
รูปที่ 6
	Residual MLP จำนวน 4 รูปแบบใน Design Space 2×2 ได้แก่ Schedule-α 4D, Schedule-α 12D, Action-q 4D และ Action-q 12D โดยแต่ละรูปแบบมีสถาปัตยกรรม Bidirectional (tanh×0.3, Δ∈[−0.3, +0.3]) และฝึกด้วย Duration Randomization ช่วง [1.5, 5.0] วินาที (V2)
ผลลัพธ์
	Action-q Residual (ทั้ง 4D และ 12D) กำจัด Velocity Reversal ได้ทุกกรณี (0/6 Reversal ทุก Duration) เทียบกับ Smoothstep ที่มี 3–4/6 Reversal โดย Action-q 12D ให้ vx_min_trans = +0.302 m/s และ Δvx_trans = 0.121 m/s
	พบ Trade-off ที่ชัดเจนใน Design Space: Action-q ชนะด้าน Safety (Reversal) แต่มี jerk_TRANS สูงกว่า Smoothstep ประมาณ 40% ส่วน Schedule-α ลด Velocity Drop ได้โดยมี Jerk ใกล้เคียง Baseline — ไม่มีวิธีใดชนะทุกตัวชี้วัดพร้อมกัน

1.4 ความต้องการของระบบ (Requirements)
1.	ระบบต้องฝึกและรัน Policy บน Isaac Lab 0.36.3 / Isaac Sim 4.5.0 กับ Unitree B1 URDF (12 DOF, ~50 กก.)
2.	ระบบต้องรองรับการเปลี่ยนรูปแบบการเดินระหว่าง Trot, Bound และ Pace ใน Duration ช่วง [1.5, 5.0] วินาที โดยวิธีที่เสนอต้องลด Velocity Reversal เทียบกับ Smoothstep Baseline

1.5 ขอบเขตของงานวิจัย (Scopes)
1.	ทดสอบเฉพาะบน Simulator (Isaac Lab) บนพื้นราบ ไม่รวมการทดสอบบนหุ่นยนต์จริงหรือภูมิประเทศขรุขระ
2.	ศึกษา Design Space 2×2 ได้แก่ Schedule Residual (α-space) และ Action Residual (q-space) ขนาด 4D และ 12D รวมเฉพาะ 3 รูปแบบการเดิน (Trot, Bound, Pace) โดยไม่รวม Steer เนื่องจากปัญหา Out-of-Distribution

1.6 ข้อกำหนดของงานวิจัย (Assumptions)
1.	Base Gait Policy แต่ละตัวถูก Freeze ระหว่าง Phase 2 — Residual MLP ไม่สามารถแก้ไข Policy ต้นทางหรือปลายทางได้โดยตรง
2.	สภาพแวดล้อมการฝึกใช้พื้นราบที่สมบูรณ์แบบ ไม่มี Domain Randomization ของภูมิประเทศ (มีเฉพาะ Seed-based Randomization ของสถานะเริ่มต้น)

 
1.7 ขั้นตอนการดำเนินงาน 
โครงงานนี้แบ่งการดำเนินงานออกเป็น 5 ขั้นตอนหลัก ดังนี้
ขั้นตอนที่ 1 คือการเตรียมสภาพแวดล้อมการจำลอง โดยติดตั้ง Isaac Lab 0.36.3 / Isaac Sim 4.5.0 และตั้งค่าหุ่นยนต์ Unitree B1 ให้สามารถใช้งานกับระบบฝึกแบบ Reinforcement Learning ได้ รวมถึงการตรวจสอบชื่อข้อต่อ จุดสัมผัสเท้า ค่า actuator stiffness/damping และตำแหน่งเริ่มต้นของหุ่นยนต์ เพื่อให้สภาพจำลองไม่เกิดความผิดพลาดพื้นฐาน เช่น เท้าจมพื้นหรือแรงสัมผัสผิดตำแหน่ง
ขั้นตอนที่ 2 คือการทดลองแนวทางเริ่มต้นด้วย CPG-RBF ร่วมกับ PI^BB Optimizer เพื่อสร้างรูปแบบการเดินแบบมีโครงสร้างจังหวะ อย่างไรก็ตาม หลังจากแก้ไขข้อผิดพลาดของสภาพแวดล้อมและฝึกใหม่ พบว่าวิธีนี้ไม่สามารถสร้างการเดินที่เสถียรบน Unitree B1 ได้เพียงพอ จึงสรุปว่าแนวทางดังกล่าวมีข้อจำกัดเชิงโครงสร้างสำหรับหุ่นยนต์หนักและมีความไม่สมมาตรของขา
ขั้นตอนที่ 3 คือการฝึก Base Gait Policies ด้วย PPO Velocity-Tracking สำหรับรูปแบบการเดิน Trot, Bound และ Pace โดยแต่ละ Policy ถูกฝึกแยกกันบนพื้นราบ และใช้ Joint Position Offset เป็น Action Space เมื่อฝึกเสร็จแล้ว Policy ทั้งหมดจะถูก Freeze เพื่อใช้เป็นต้นทางและปลายทางของการเปลี่ยนรูปแบบการเดินใน Phase 2
ขั้นตอนที่ 4 คือการออกแบบและฝึก Residual MLP ใน Design Space 2×2 ประกอบด้วย Schedule Residual (แก้ไข Blending Coefficient α) และ Action Residual (แก้ไข Joint Target โดยตรง) แต่ละแบบในขนาด 4D (per-leg) และ 12D (per-joint) รวม 4 รูปแบบ ทุกรูปแบบใช้ Bidirectional Clamp (tanh×0.3, Δ∈[−0.3, +0.3]) เพื่อให้ MLP สามารถชะลอหรือเร่ง Transition ได้ และฝึกด้วย Duration Randomization ช่วง [1.5, 5.0] วินาที เพื่อความ Robust ข้าม Duration
ขั้นตอนที่ 5 คือการประเมินผลและเปรียบเทียบ โดยใช้ตัวชี้วัดหลักคือ vx_min_trans (ความเร็วต่ำสุดในช่วง Transition), Δvx_trans (การลดลงของความเร็วเทียบกับ Pre-transition), jerk_TRANS, Reversal Rate และ CoT รวมถึงทำ Duration Sweep ที่ Duration 1.5–5.0 วินาที เพื่อทดสอบความ Robust ของแต่ละวิธีข้าม Duration

 
บทที่ 2 ทฤษฎี/งานวิจัย/การศึกษาที่เกี่ยวข้อง
โครงงานนี้ผสานแนวคิดจากหลายสาขาเข้าด้วยกัน บทนี้อธิบายเฉพาะทฤษฎีที่มีบทบาทโดยตรงต่อการออกแบบระบบ ได้แก่ PPO ซึ่งใช้ฝึก Base Gait Policy ทั้ง 4 แบบใน Phase 1, Residual Policy Learning ซึ่งเป็นรากฐานสถาปัตยกรรมของ Phase 2, โครงสร้างการประสานงานของขา (Gait Coordination) ซึ่งอธิบายว่าทำไม Per-Leg Residual จึงจำเป็น, Smoothstep ซึ่งใช้เป็น Baseline Schedule ที่ MLP เรียนรู้ค่าแก้ไขบน และสถาปัตยกรรมของ Unitree B1 ซึ่งกำหนด Constraint ด้านน้ำหนักและความไม่สมมาตรของขาที่ส่งผลต่อทุกการตัดสินใจออกแบบ

2.1 Reinforcement Learning และ Proximal Policy Optimization
Reinforcement Learning (RL) เป็นกรอบการเรียนรู้ที่ Agent ทำปฏิสัมพันธ์กับ Environment โดยเลือก Action จาก State ปัจจุบัน เพื่อให้ได้ Cumulative Reward สูงสุด ปัญหาถูกสร้างแบบจำลองเป็น Markov Decision Process (MDP) กำหนดด้วย (S, A, P, R, γ) โดย S คือ State Space, A คือ Action Space, P คือ Transition Probability, R คือ Reward Function และ γ คือ Discount Factor
2.1.1 Proximal Policy Optimization (PPO)
	PPO (Schulman et al., 2017) เป็นอัลกอริทึม Policy Gradient แบบ On-Policy ที่จำกัดการเปลี่ยนแปลง Policy ในแต่ละ Update ผ่าน Clipped Surrogate Objective: L(\theta)\ =\ min(r(\theta)Â,clip(r(θ),1-ε,1+ε)Â) โดย r(\theta) คืออัตราส่วนความน่าจะเป็นระหว่าง Policy ใหม่และเก่า, Â คือ Advantage Estimate และ \varepsilon เป็นค่า Clipping (โครงงานนี้ใช้ \varepsilon = 0.2) การจำกัดนี้ป้องกัน Policy จากการเปลี่ยนแปลงอย่างรุนแรงในแต่ละ Iteration ซึ่งมีความสำคัญอย่างยิ่งสำหรับงานควบคุมหุ่นยนต์ที่ Policy ที่ไม่เสถียรอาจทำให้เกิดการล้ม
	โครงงานนี้ใช้ PPO ผ่าน RSL-RL OnPolicyRunner ทั้ง Phase 1 (ฝึก Base Gait Policy 4 แบบ แต่ละแบบ 12-D Joint Position Offset) และ Phase 2 (ฝึก Residual MLP 4-D Per-Leg Δα) โดยใช้ 4,096 Environment แบบขนาน Rollout Length 24 Steps, Mini-batch Size 96 และ Learning Rate 1\times{10}^{-3}\ ด้วย Generalized Advantage Estimation (GAE, \lambda=0.95)

2.2 Residual Policy Learning
Residual Policy Learning (Silver et al., 2018; Johannink et al., 2018) เป็นแนวทางที่ให้ Neural Network เรียนรู้ค่าแก้ไข (Residual) บน Baseline Controller แทนที่จะเรียนรู้ Policy ทั้งหมดตั้งแต่ต้น ข้อดีหลักคือ (1) Baseline ให้พฤติกรรมที่สมเหตุสมผลตั้งแต่เริ่มต้น ลด Exploration ที่อันตราย (2) Residual ถูกจำกัดขนาดได้ ทำให้มี Safety Bound ที่ชัดเจน (3) การตั้งค่า Residual เป็นศูนย์จะกลับสู่ Baseline ทันที ทำให้เปรียบเทียบ Counterfactual ได้โดยตรง ในโครงงานนี้ Baseline คือ Smoothstep Schedule (\alpha_baseline\ =\ x²(3-2x)) และ Residual MLP เรียนรู้ค่า \mathrm{\Delta\alpha}\ \in\ [0,\ 0.3] แยกแต่ละขา เพิ่มบน Baseline เพื่อปรับจังหวะการเปลี่ยนรูปแบบการเดิน
2.3 Gait Coordination ของหุ่นยนต์สี่ขา
รูปแบบการเดินของหุ่นยนต์สี่ขากำหนดโดยโครงสร้างการประสานงานของขาคู่ (Leg-Pair Coordination Structure) สัตว์สี่ขาในธรรมชาติใช้รูปแบบการเดินหลากหลาย ได้แก่ Walk (เดินช้า ขาสัมผัสพื้นทีละข้าง), Trot (ขาคู่ทแยง), Bound (ขาคู่หน้า-หลัง), Pace (ขาคู่ด้านข้าง), Gallop (ขาเคลื่อนเหลื่อมเวลา) และ Pronk (ขาทั้งสี่พร้อมกัน) เป็นต้น โครงงานนี้เลือกศึกษา 3 รูปแบบที่มีโครงสร้างการประสานงานแตกต่างกันอย่างชัดเจน ได้แก่ 
	Trot ขาคู่ทแยง (FL+RR, FR+RL) สลับจังหวะเป็นคู่



	Bound — ขาคู่หน้า-หลัง (FL+FR, RL+RR) เคลื่อนพร้อมกัน



	Pace — ขาคู่ด้านข้าง (FL+RL, FR+RR) เคลื่อนพร้อมกัน 

ทั้งสามรูปแบบนี้ครอบคลุมทุก Coordination Axis (Diagonal, Fore-aft, Lateral) ทำให้เป็นชุดทดสอบที่เพียงพอสำหรับการศึกษา Gait Transition ความแตกต่างเชิงโครงสร้างนี้สามารถสังเกตได้ชัดเจนจากพฤติกรรม เช่น เมื่อเปลี่ยนจาก Trot เป็น Bound ขา FL ต้องเลิกประสานกับ RR (คู่ทแยง) แล้วไปประสานกับ FR (คู่หน้า) แทน ซึ่งต้องการค่า α ที่แตกต่างกันในแต่ละขาชั่วคราว นี่คือข้อเหตุผลเชิงสถาปัตยกรรมของ Per-Leg Residual Structure
	Smoothstep Interpolation

 
Smoothstep เป็น Hermite Polynomial ลำดับ 3
 S(x)\ =\ x²(3-2x) สำหรับ x ∈ [0,1] 
มีคุณสมบัติ
dS/dx\ =\ 0 ที่จุดปลายทั้งสอง (x=0 และ x=1)
ซึ่งขจัด Kinematic Kick ที่จุดเริ่มต้นและสิ้นสุดของ Transition ที่ Linear Ramp (α = x) มี \frac{d\alpha}{dt}\ \neq\ 0 ที่จุดปลาย ทำให้เกิดความไม่ต่อเนื่องของความเร่งเชิงมุม 
ในโครงงานนี้ Smoothstep ถูกใช้เป็น Baseline Schedule สำหรับ α โดย x\ =\ clamp(\frac{t\ -\ t_start}{duration\ }\ ,\ 0,\ 1) แล้ว \alpha_baseline\ =\ x²(3-2x) Residual MLP จะเพิ่มค่า \mathrm{\Delta\alpha}\ \in\ [0,\ 0.3] บน Baseline นี้
2.5 สถาปัตยกรรมหุ่นยนต์ Unitree B1
Unitree B1 เป็นหุ่นยนต์สี่ขาขนาดกลาง น้ำหนัก ~50 กก. มี 12 องศาอิสระ (3 ข้อต่อต่อขา: Hip Abduction, Thigh Flexion, Calf Extension) มีความไม่สมมาตรเชิงโครงสร้าง (Morphological Asymmetry) ที่สำคัญ คือมุม Default ของ Thigh Joint ด้านหน้า (0.8 rad) ต่างจากด้านหลัง (1.0 rad) ขนาด 0.2 rad ความไม่สมมาตรนี้ทำให้ขาหน้าและขาหลังต้องการอัตราการเปลี่ยน α ที่แตกต่างกันระหว่าง Transition ซึ่งเป็นแรงจูงใจโดยตรงของ Per-Leg Residual Structure ค่า Actuator ที่ใช้ Stiffness 400 N·m/rad, Damping 10 N·m·s/rad, Action Scale 0.25
 
2.6 งานวิจัยที่เกี่ยวข้อง
ส่วนนี้อธิบายว่างานวิจัยก่อนหน้าแต่ละงานส่งผลต่อการตัดสินใจออกแบบในโครงงานนี้อย่างไร โดยไม่ได้เป็นการสรุปงานวิจัยทั่วไป

(1) Thor, Kulvicius & Manoonpong (2021) — IEEE Transactions on Neural Networks and Learning Systems 32(9), 4013–4025
งานนี้นำเสนอ Generic Neural Locomotion Control Framework สำหรับหุ่นยนต์ขาหลาย Morphology ซึ่งสามารถนำไปใช้กับหุ่นยนต์ที่มีรูปร่างแตกต่างกันได้โดยไม่ต้องออกแบบ Controller ใหม่ Framework ประกอบด้วยสองส่วนหลัก ได้แก่ CPG-RBF Network และ {PI}^{BB} Black-Box Optimizer
CPG-RBF Network: SO(2) Oscillator คู่ทำหน้าที่เป็น Central Pattern Generator สร้างสัญญาณ Phase\ (sin,\ cos) ด้วยความถี่คงที่ ~0.3 Hz RBF Network รับสัญญาณนี้และ Map ผ่าน Radial Basis Functions ไปยัง Joint Trajectory ทำให้สามารถสร้าง Arbitrary Rhythmic Trajectory โดยปรับเฉพาะน้ำหนัก W ใน RBF Layer (Matrix W ขนาด 20\times3 รวม ~60 Parameter) เมื่อเทียบกับ PPO ที่มี ~100,000 Parameter 
ข้อดีคือ มีความ Generalizability ข้ามหุ่นยนต์หลาย Morphology, Fast Learning ภายในไม่กี่ร้อย Iteration, Interpretability เพราะ W มีขนาดเล็กและ Map ตรงกับ Trajectory ที่สังเกตได้
{PI}^{BB} Optimizer เป็น Black-Box Optimization ที่ไม่ต้องคำนวณ Gradient โดย Sample W จาก Gaussian Distribution แล้วรัน Episode เพื่อประเมิน Reward ของแต่ละ Sample จากนั้น อัปเดต Distribution ใหม่โดยถ่วงน้ำหนักตาม Reward ผ่าน Softmax Weighting (หลักการเดียวกับ {PI}^2ของ Theodorou et al.) กลไกนี้ทำงานได้ดีบนหุ่นยนต์เบา เช่น Hexapod ที่ทดลองในงานต้นฉบับ เพราะมี Reward Spread ระหว่าง Sample โดย Sample ที่เดินได้จะได้ Reward สูง แต่ Sample ที่ล้มจะได้ Reward ต่ำ การใช้ Softmax Weighting จึงมีทิศทางที่ชัดเจน อย่างไรก็ตาม บน Unitree B1 (~50 กก.) เกือบทุก Perturbation สามารถทำให้ล้มได้ จาก Reward Spread ที่ใกล้ศูนย์ Softmax Weights จึงเท่ากันทุก Sample และการ Update กลายเป็น Noise-Weighted Average ที่ W ไม่เรียนรู้
บทบาทในโครงงาน: โครงงานนี้ทดลอง CPG-RBF + {PI}^{BB} บน Unitree B1 ใน Phase 1 ผ่าน 12 Encoding Experiment และ Fix Bug 5 ข้อ (Action Scale, Stiffness, Spawn Height, Dead Reward Code, Air-Time Variance) ผลลัพธ์สุดท้ายหลัง Full Retrain: vx = +0.091 m/s (Oscillatory Lunge, std = 0.171) เทียบกับ PPO ที่ +0.434 m/s — ช่องว่าง 4.8 เท่า การวิเคราะห์พิสูจน์ข้อจำกัดเชิงโครงสร้าง 3 ประการ
	PI^BB Reward Collapse บน B1
	(ii) Shared W ไม่รองรับ Thigh Asymmetry 0.2 rad (Front 0.8 rad, Rear 1.0 rad)
	"Lunge-and-Fall" เป็น Local Optimum ที่ PI^BB ไม่สามารถหลุดได้ การพิสูจน์ว่าข้อจำกัดเหล่านี้เป็น Structural ไม่ใช่ Bug ยืนยันว่าการ Pivot ไปใช้ PPO ถูกต้อง และ Phase 2 มี Contribution ที่ไม่ขึ้นกับแนวทาง Base Policy

 
รูปที่ 7 CPG-RBF network, combining a CPG with an RBF network. The policy is encoded in the synaptic weights, wpj,k , connecting the RBF neurons, Rh , to the motor neurons, Mj . These weights are optimized using BBO as indicated by the dashed arrow. The weights w0,0, w0,1, w1,0, and w1,1 of the CPG are fixed such that its outputs oscillate at a certain frequency [i.e., here approximately 0.3 Hz (low walking frequency)].

(2) Rostro-Gonzalez et al. (2025) — Biomimetics 10(6), 381 / Shafiee, Bellegarda & Ijspeert (2024) — Nature Communications 15, 3073
งานทั้งสองนี้ตอบคำถามที่แตกต่างกันแต่เสริมกัน: Rostro-Gonzalez ตอบว่า เมื่อไรควรเปลี่ยน Gait (When) ส่วน Shafiee ตอบว่า ทำไมถึงต้องเปลี่ยน (Why)
Rostro-Gonzalez et al.: งานนี้ใช้ Spiking Neural Network (SNN) เป็น CPG สำหรับสร้างรูปแบบการเดิน 3 แบบ (Walk, Jog, Run) บน Hexapod จริงที่มี FSR Sensor บนขา SNN สร้าง Spike Train ที่ประสานงานกันระหว่างขา กลไกสำคัญคือ SPIKE-synchronization Metric — วัดความสอดคล้องของ Spike Train ระหว่างขาคู่ที่ต้องการ Synchronize เพื่อค้นหาช่วงเวลาที่ขาอยู่ใน Phase ที่เหมาะสมที่สุดสำหรับ Transition ในเชิงคณิตศาสตร์ SPIKE-sync วัดว่า Spikes ของ Neuron สองตัวมีความใกล้ชิดกันในเวลาพอที่จะถือว่า "Synchronous" หรือไม่ เมื่อ Synchrony ถึงระดับ Threshold ระบบจะเปิด Transition ผลลัพธ์: Transition แทบไม่สังเกตได้ (Imperceptible) ทดสอบบนพื้น 4 ประเภท และ Mean Stepping Time Error ต่ำกว่าการเปลี่ยนแบบ Discrete Switch อย่างมีนัยสำคัญ
Shafiee et al.: งานนี้ฝึก Reinforcement Learning Policy บน Unitree A1 ให้เรียนรู้ที่จะเปลี่ยน Gait อัตโนมัติโดยใช้ Viability เป็น Criterion ไม่ใช่ Energy Efficiency เพียงอย่างเดียว Viability หมายถึงความสามารถในการหลีกเลี่ยงการล้มใน Future Trajectory ซึ่ง Policy เรียนรู้ได้จาก Reward ที่ลงโทษ Fall อย่างรุนแรง ผลลัพธ์: A1 สามารถเปลี่ยน Trot→Pronk โดยอัตโนมัติเพื่อข้ามช่องว่าง 30 cm (83.3% of body length) ที่ความเร็ว >1.3 m/s โดยไม่มีการ Pre-program Transition Trigger งานนี้แสดงว่า Energy Hypothesis เพียงอย่างเดียวไม่เพียงพออธิบาย Gait Transition ของสัตว์ — Injury Avoidance และ Fall Prevention เป็นแรงขับที่สำคัญกว่า
บทบาทในโครงงาน: Rostro-Gonzalez สนับสนุน Time-Gating ในสถาปัตยกรรม Phase 2 — งานนี้แสดงว่าจังหวะการเปลี่ยน (When) มีความสำคัญเชิงชีวกลศาสตร์ ซึ่งเป็นเหตุผลที่ Time-Gate บังคับ Δα = 0 นอก Transition Window แทนที่จะปล่อยให้ MLP ตัดสินใจเอง Shafiee สนับสนุน Asymmetric Clamp [0, 0.3] — บน B1 (50 กก.) การล้มระหว่าง Transition เป็นอันตรายจริง Velocity Reversal (vx < 0) เป็น Precursor ของ Fall การที่ Asymmetric Clamp ป้องกันไม่ให้ α ตกต่ำกว่า Smoothstep Baseline เป็น Structural Viability Constraint อย่างหนึ่ง
 
รูปที่ 8 SPIKE- synchronization: (a) general scheme and (b) example of transition from jogging to running at time 
(3) Ubellacker, Csomay-Shanklin, Molnar & Ames (2022) — IROS
งานนี้นำเสนอ Motion Primitive Graph สำหรับวางแผน Transition ระหว่าง Dynamic Motion Primitives บน Quadruped อย่างปลอดภัย แนวคิดหลักคือการ Formalize ว่า Transition ระหว่าง Primitive ปลอดภัยหรือไม่ โดยพิจารณา Dynamic State ณ ขณะ Transition ไมใช่แค่ Pose State
Control Barrier Functions: เป็นเครื่องมือทางคณิตศาสตร์ที่รับประกันว่า System State จะไม่ออกจาก Safe Set CBF กำหนดฟังก์ชัน h(x) >= 0 ที่แสดง Safe Region และสังเคราะห์ Controller ที่รักษา h(x) >= 0 ตลอดเวลา สำหรับ Legged Robot Safe Set อาจหมายถึง ขาที่ยืนไม่ลื่น หรือ Momentum ไม่เกิน Threshold ที่จะทำให้ล้ม
Motion Primitive Graph: แต่ละ Node คือ Motion Primitive (Lie, Stand, Walk, Jump) ที่มี Safe Region of Attraction S(t) แต่ละ Edge คือ Transition ที่ผ่านการตรวจสอบแล้วว่าหาก System อยู่ใน S(t) ของ Primitive ต้นทาง การ Transition ไปยัง Primitive ปลายทางจะยังคงอยู่ใน Safe Set ของปลายทาง ทดสอบบน Unitree A1 บน Hardware จริง
บทบาทในโครงงาน: งานนี้ Motivate Asymmetric Clamp (sigmoid x 0.3) ใน Phase 2 โดยตรง CBF บอกวา อยาออกจาก Safe Set สวน delta-alpha ใน [0, 0.3] บอกวา alpha จะไมต่ำกวา Smoothstep Baseline เดดขาด ซึ่งเปน Structural Guarantee ที่ Bake ไวในสถาปตยกรรม ความแตกตางสำคัญ: CBF ใน Ubellacker et al. เปน Formal Proof ที่ตองการ Dynamic Model สวน Asymmetric Clamp เปน Heuristic Bound ที่ทำงานไดโดยไมตองการ Model แตมีเปาหมายเดียวกันคือปองกัน Unsafe Transition

(4) Silver, Allen, Tenenbaum & Kaelbling (2018) / Johannink et al. (2019) — ICRA
Silver et al. นำเสนอ Residual Policy Learning (RPL) ครั้งแรก: ให้ RL Policy เรียนรู้เฉพาะค่าแก้ไข (Residual) บน Hand-Engineered Controller ไม่ใช่เรียนรู้ Policy ใหม่ทั้งหมดตั้งแต่ต้น งานต้นฉบับทดลองใน Block Assembly ในโลกจริง — Hand-Engineered Controller จัดการ Approach/Grasp ได้ดีในกรณีทั่วไป แต่ล้มเหลวเมื่อ Object มี Orientation ที่หลากหลาย RL Residual Policy เรียนรู้เฉพาะ Correction ส่วนที่ Controller พื้นฐาน 'ผิด' ผ่าน Real-World Experience โดยตรง
ข้อดีหลักของ RPL: (i) Policy ขนาดเล็กเรียนรู้เร็ว เพราะ Search Space ลดเหลือเฉพาะ Correction Space (ii) Safety โดย Default — ถ้า Residual เข้าใกล้ศูนย์ ระบบ Fallback ไปยัง Baseline ที่ทำงานได้ (iii) Interpretability — ขนาดของ Residual บอกตรงๆ ว่า Policy ไว้วางใจ Baseline มากแค่ไหน Johannink et al. ขยาย RPL ไปยัง Real Robot Arm Control บน Hardware แสดงว่า Residual สามารถ Bridge Sim-to-Real Gap ได้ เพราะ Baseline ถูก Tune สำหรับ Real Environment และ RL เรียนรู้เฉพาะ Correction สำหรับ Mismatch
บทบาทในโครงงาน: Phase 2 ทั้งหมดตั้งอยู่บน RPL แต่มีความแตกต่างสำคัญ 3 ประการ: (i) Baseline คือ Smoothstep Schedule ซึ่งเป็นฟังก์ชันคณิตศาสตร์ h(t) ไม่ใช่ Demo Policy หรือ Hand-Engineered Controller — เป็น Baseline ที่กำหนด Transition Timing ไว้แล้ว (ii) Residual แยกต่อขา 4 ค่า (delta-alpha FL, FR, RL, RR) ไม่ใช่ Output Space เดียว เพื่อรองรับ Asymmetric Coordination Change ระหว่าง Gait (iii) Asymmetric Clamp [0, 0.3] กำหนดทิศทางของ Residual ว่า MLP สามารถเร่ง Transition ได้เท่านั้น ไม่สามารถชะลอ ซึ่งปิด Delay-Rush Exploit ที่พบในเวอร์ชัน Symmetric tanh x 0.8
 
รูปที่ 9 ) Silver et al. (2018) trains an agent directly in the real world to solve a model assembly task involving contacts and unstable objects. An outline of their method, which consists of combining hand-engineered controllers with a residual RL controller, is shown on the left. Rollouts of residual RL solving the block insertion task are shown on the right. Residual RL is capable of learning a feedback controller that adapts to variations in the orientations of the standing blocks and successfully completes the task of inserting a block between them. Videos are available at . residualrl.github.io
(5) Rudin, Hoeller, Reist & Hutter (2022) — CoRL / Siekmann et al. (2021) — RSS
Rudin et al. นำเสนอการฝึก PPO Policy สำหรับ Quadruped Locomotion ด้วย Massively Parallel Training — 4,096 Robot Instance พร้อมกันบน GPU เดียว ทำให้เรียนรู้การเดินบน Unitree Go2 ได้ภายในไม่กี่นาที แนวคิดสำคัญ: (i) IsaacGym Parallelism ทำให้ Sample Efficiency สูงมาก (ii) Reward Stack ที่ Calibrate ไว้สำหรับ Legged Locomotion โดยเฉพาะ ได้แก่ Velocity Tracking Reward ที่ Tight (std = 0.25), Foot Air-Time Reward, Action Rate Penalty, Foot Clearance Reward (iii) Curriculum Randomization ค่อยๆ เพิ่ม Terrain Difficulty RSL-RL Framework ที่ใช้ในงานนี้กลายเป็น Standard สำหรับ Isaac Lab-based Locomotion ทั่วโลก
Siekmann et al. แสดง Joint Position Offset Action Space สำหรับ Bipedal Locomotion — Policy Output คือ Offset จาก Default Joint Pose (ไม่ใช่ Absolute Position) การออกแบบนี้ทำให้ Policy เรียนรู้ได้ง่ายขึ้นเพราะ Action Space มีศูนย์กลางที่ตำแหน่ง 'ยืนนิ่ง' ซึ่งปลอดภัยเสมอ และ Policy ต้องเรียนรู้เฉพาะส่วนที่เพิ่มเติมจาก Default Scale Factor 0.25 (ใช้ใน Phase 1 ของโครงงานนี้) จำกัด Action Range เพื่อป้องกัน Joint Limit Violation
บทบาทในโครงงาน: Engineering Pattern ทั้งหมดของ Phase 1 มาจากสองงานนี้: (i) Parallel Training ด้วย 4,096 Env ผ่าน RSL-RL OnPolicyRunner (ii) Joint Position Offset x 0.25 ตาม Siekmann et al. (iii) Reward Stack พื้นฐาน ซึ่งโครงงานนี้ปรับเพิ่ม 11 Custom Term เฉพาะสำหรับ B1 ได้แก่ Base Height L2 (target 0.42 m), Air-Time Variance Penalty, Short Swing Penalty, LR Symmetry Penalty, Duty Factor Target, True Bound/Pace Reward และอื่นๆ การปรับเหล่านี้จำเป็นเพราะ B1 (~50 กก.) มี Failure Mode ที่ต่างจาก Go2 (~15 กก.) อย่างมาก เช่น Standstill Exploit, Crawling Exploit (Body สาก 0.18 m) และ 2-Leg Pathology

 
บทที่ 3 ระเบียบวิธีวิจัย

บทนี้อธิบายระเบียบวิธีวิจัยแบ่งเป็น 2 ระยะ: Phase 1 — ฝึก Base Gait Policy ด้วย PPO สำหรับ Trot, Bound และ Pace บนพื้นราบ และ Phase 2 — ฝึก Per-Leg Residual MLP สำหรับการเปลี่ยนรูปแบบการเดินอย่างราบรื่น รวมถึงรายละเอียด Observation Space, Reward Function, สถาปัตยกรรม Network และวิธีการประเมินผล

	แนวทางเริ่มต้นและเหตุผลในการปรับเปลี่ยน
	แนวทางแรกที่ทดลองคือ CPG-RBF (Central Pattern Generator + Radial Basis Function) ร่วมกับ PI^BB Optimizer (Thor et al., 2021) ซึ่งใช้ SO(2) Oscillator สร้างสัญญาณจังหวะและ RBF Network แปลงเป็นคำสั่ง Joint โดยทดลองทั้งหมด 12 การทดลองเข้ารหัส (Encoding Experiments) ตลอดสัปดาห์ที่ 10–11 ของโครงงาน
	หลังจากพบและแก้ไข Bug 5 จุดในระบบ (Action Scale ขาด ×0.25, Stiffness ผิด 200→400, ความสูง Spawn ผิด 0.42→0.50 m, Reward Code ที่ไม่ถูกเรียก และ Air-time Variance Penalty ที่ขาดหายไป) ได้ทำการทดสอบเชิงตรวจสอบ 2 แบบ:
	Experiment A — น้ำหนักเดิมในสิ่งแวดล้อมที่แก้ไขแล้ว: หุ่นยนต์ยืนนิ่ง vx = 0.000 m/s เนื่องจากน้ำหนักถูก Train ด้วยการเคลื่อนที่ใหญ่เกิน 4 เท่า
	Experiment B — Train ใหม่ในสิ่งแวดล้อมที่แก้ไขแล้ว: ได้ vx = +0.091 m/s เท่านั้น (แบบ Oscillatory Lunge, std = 0.171) ซึ่งต่ำกว่า PPO Trot (+0.434 m/s) ถึง 4.8 เท่า
	สาเหตุเชิงโครงสร้างที่ทำให้ CPG-RBF + PIBB ไม่สามารถทำงานได้กับ B1 มี 3 ประการ: (1) PIBB ล้มเหลวบนหุ่นยนต์หนัก — เมื่อหุ่นยนต์ 50 กก. แทบทุกการสำรวจทำให้ล้ม Softmax Weights จึงเท่ากันหมดและ W หยุดเรียนรู้ (2) W ที่ใช้ร่วม (Shared) ไม่สามารถรองรับความไม่สมมาตรของ B1 ที่มีค่า Thigh ต่าง 0.2 rad ระหว่างขาหน้าและขาหลัง และ (3) Reward Landscape มี Local Optimum ที่ "พุ่งไปข้างหน้าแล้วล้ม" ได้คะแนนดีกว่า Cyclic Gait เสมอ
	การเปลี่ยนไปใช้ PPO จึงไม่ใช่การละทิ้ง แต่เป็นผลจากการพิสูจน์อย่างเป็นระบบว่า PIBB มีข้อจำกัดเชิงโครงสร้างที่ไม่สามารถแก้ไขด้วยการปรับ Hyperparameter ได้

3.2 แนวทางที่เลือกใช้ (Per-leg Residual Learning)
จากข้อจำกัดของ CPG-RBF จึงเลือกแนวทาง Two-Phase Design ตามแนวคิด Residual Policy Learning (Silver et al., 2018): Phase 1 ฝึก Base Gait Policy ด้วย PPO สำหรับแต่ละรูปแบบการเดิน และ Phase 2 Freeze Base Policies แล้วฝึก Per-Leg Residual MLP ขนาดเล็กที่เรียนรู้ค่าแก้ไข Δα ∈ [0, 0.3] แยกแต่ละขา เพิ่มบน Smoothstep Baseline Schedule ข้อดีของสถาปัตยกรรมนี้คือ: (1) Base Policies ทำงานโดยไม่ถูกแก้ไขในช่วง Steady-State (2) MLP แก้ไขเฉพาะช่วง Transition เท่านั้นผ่าน Time-Gating (3) Asymmetric Clamp ป้องกันไม่ให้ α ต่ำกว่า Baseline และ (4) Per-Leg Structure รองรับการเปลี่ยนคู่ประสานงานที่แตกต่างกันของแต่ละขา
3.3 Phase 1 — Base Gait Policies	
ฝึก PPO Velocity-Tracking Policy จำนวน 4 แบบ (Trot, Bound, Pace, Steer) บนพื้นราบ ด้วย Isaac Lab Manager-Based RL และ RSL-RL OnPolicyRunner แต่ละ Policy รับ Observation 45 มิติ (Base Velocity 6D, Projected Gravity 3D, Joint Position 12D, Joint Velocity 12D, Last Action 12D) และส่งออก Action 12 มิติ (Joint Position Offset ×0.25 บวกกับ Default Joint Pose) ฝึกด้วย 4,096 Environment แบบขนาน, Learning Rate 1×10⁻³, Discount Factor γ=0.99, GAE λ=0.95 บนหุ่นยนต์ Unitree B1 ที่ปรับค่า Actuator Stiffness เป็น 400 N·m/rad สำหรับรองรับน้ำหนัก 50 กก.
3.3.1 Custom Reward Engineering สำหรับ B1
	ปัญหาเฉพาะของ B1 ที่ต้องแก้ด้วย Reward Term เพิ่มเติม: Standstill Local Optimum (แก้โดย Tighten Velocity Tracking std 0.5→0.25), Crawling Exploit (แก้โดย Base Height L2 Penalty เป้าหมาย 0.42 m), 2-Leg Trot Pathology (แก้โดย Excessive Air/Contact Time Penalty), Bilateral Asymmetry (แก้โดย Joint L/R Symmetry Penalty) รวม 11 Custom Reward Term ใน b1_velocity_mdp.py
	สำหรับ Phase 2 ใช้เฉพาะ 3 Gait (Trot, Bound, Pace) โดยตัด Steer ออกเนื่องจากฝึกที่ yaw ∈ (0.4, 1.0) ซึ่ง Out-of-Distribution ที่ yaw=0 ของ Phase 2 คุณภาพ Base Policy วัดจาก: Trot vx +0.434 m/s, Bound vx +0.5 m/s, Pace vx +0.45 m/s ทุก Policy มี Zero-Fall Rate บนพื้นราบ

3.4 Phase 2 — Design Space 2×2 ของ Residual Transition Learning
Freeze 3 Base Policy (Trot, Bound, Pace) แล้วฝึก Residual MLP ใน Design Space 2 มิติ ได้แก่ (1) Correction Space: Schedule Residual ที่แก้ไข Blending Coefficient α เพื่อควบคุมจังหวะการเปลี่ยน และ Action Residual ที่แก้ไข Joint Target โดยตรงหลังการ Blend และ (2) Resolution: 4D (per-leg) หรือ 12D (per-joint) รวมเป็น 4 รูปแบบ
สถาปัตยกรรม V2 มีการปรับปรุงหลักจาก V1 ดังนี้: (i) Bidirectional Clamp: เปลี่ยนจาก sigmoid×0.3 (Δ∈[0,0.3], เร่งได้อย่างเดียว) เป็น tanh×0.3 (Δ∈[−0.3,+0.3], ชะลอหรือเร่งได้) (ii) Duration Randomization: ฝึกด้วย Transition Duration สุ่มช่วง [1.5, 5.0] วินาที ต่อ Episode เพื่อให้ Policy เรียนรู้ Strategy ทั่วไป (iii) Policy-Phase Observation: เพิ่ม π_current (12D) + π_target (12D) ลงใน Observation เพื่อให้ MLP สังเกต Output ของ Policy ต้นทางและปลายทาง ณ ขณะนั้น และ (iv) Reward ปรับใหม่: แทน Joint Jerk Penalty (ซึ่งมีผลน้อยมาก) ด้วย vx_window Penalty (−2.0) ที่ลงโทษความเร็วต่ำโดยตรงในช่วง Transition Window
3.5 Blending และ Time-Gating
ในแต่ละ Control Step (50 Hz): MLP รับ Observation 70D (4D variants) หรือ 78D (12D variants) ประกอบด้วย Base State 45D, norm_duration 1D, π_current 12D, π_target 12D และสำหรับ 12D variants เพิ่ม Last Action 8D ส่งออก Residual ดิบผ่าน tanh×0.3 ได้ Δ∈[−0.3,+0.3] จากนั้น Time-Gating บังคับให้ Residual = 0 นอกช่วง Transition Window
สำหรับ Schedule Residual (α-space): α_k = clamp(α_baseline + Δα_k, 0, 1) ต่อขา k แล้ว blended_k = (1−α_k)·π_current[k] + α_k·π_target[k]
สำหรับ Action Residual (q-space): blend ด้วย α_baseline ก่อน แล้วบวก Δq: joint_target_k = blended_k + Δq_k
ทั้งสองกรณีใช้ joint_target = default_pose + 0.25 × joint_target สุดท้าย
	3.5.1 ความแตกต่างเชิงสถาปัตยกรรมระหว่าง Schedule และ Action Residual
	Schedule Residual (Δα) ทำงานใน Timing Space: ควบคุมว่าแต่ละขาควรเปลี่ยนจาก Gait ต้นทางไปยัง Gait ปลายทางเร็วหรือช้าเพียงใด เหมาะกับสมมติฐานว่าปัญหาหลักของ Gait Transition คือ Phase Mismatch ระหว่าง Policy ต้นทางและปลายทาง ข้อได้เปรียบคือ Search Space เล็ก (4D หรือ 12D) และตีความได้ง่าย แต่ข้อจำกัดคือสามารถแก้ได้เฉพาะ “เมื่อไร” ไม่ใช่ “เท่าไร” ของ Joint Position
Action Residual (Δq) ทำงานใน Joint Space: บวก Correction โดยตรงบน Joint Target หลังการ Blend เหมาะกับสมมติฐานว่าปัญหาหลักคือ Joint Target ที่ Blend แล้วยังไม่ถูกต้องในเชิงตำแหน่ง ข้อได้เปรียบคือแก้ได้ทั้ง Position และ Timing พร้อมกัน แต่ข้อจำกัดคือ Search Space ใหญ่กว่าและอาจผลิต Jerk สูงถ้าไม่ถูก Constrain เพียงพอ
โครงงานนี้ทดสอบทั้งสองแนวทางเชิงประจักษ์ใน Design Space เดียวกัน เพื่อให้ผลการเปรียบเทียบตอบคำถามว่าการเลือก Correction Space มีผลต่อ Trade-off ระหว่าง Safety (Reversal) และ Smoothness (Jerk) อย่างไร

3.6 Reward Function สำหรับ Phase 2 (V2)
Reward Function ของ V2 ออกแบบให้ Incentive หลักอยู่ที่การรักษาความเร็วระหว่าง Transition โดยตรง ประกอบด้วย: Velocity Tracking (+1.5 exp, std=0.25), Yaw Tracking (+0.75 exp), Body Upright (−2.0), Body Height (−50.0 L2), Action Rate Penalty (−0.5 step-to-step change), Residual Sparsity (−0.5 L2 สำหรับ 4D; −0.167 สำหรับ 12D — normalize per-dim), vx_window Penalty (−2.0 ลงโทษ vx ต่ำในช่วง Transition Window โดยตรง), Alive Bonus (+0.5)
ความแตกต่างสำคัญจาก V1: นำ Joint Jerk Penalty (−1×10⁻¹⁰ ซึ่งมีผลน้อยมากในทางปฏิบัติ) ออก และแทนด้วย vx_window Penalty ที่ลงโทษ Velocity Drop โดยตรง ทิศทางนี้ทำให้ MLP เรียนรู้ที่จะรักษา Velocity ในช่วง Transition แทนที่จะพยายามลด Jerk ทางอ้อม
3.7 วิธีการประเมินผล
การประเมินผลแยก Transition Window ออกจากช่วง Steady-State เพื่อวัดเฉพาะผลของการเปลี่ยนรูปแบบการเดิน ตัวชี้วัดหลักที่ใช้:
	vx_min_trans: ความเร็วต่ำสุดภายใน Transition Window ของแต่ละ Gait Pair (เฉลี่ยจาก 6 คู่) — ค่าสูงหมายถึงหุ่นยนต์ไม่สูญเสีย Momentum
	Δvx_trans: ค่าเฉลี่ย Pre-transition vx ลบด้วย vx_min_trans — วัดขนาดของ Velocity Dip
	jerk_TRANS: RMS ของอัตราเปลี่ยนแปลง Joint Acceleration ใน Transition Window — วัดความรุนแรงของคำสั่งข้อต่อ
	Reversal Rate: จำนวน Gait Pair ที่ vx_min_trans < 0 (หุ่นยนต์เดินถอยหลัง) จาก 6 คู่
	CoT: Cost of Transport = Power / (mass × g × vx) เฉลี่ยตลอด Episode
การทดลองหลักใช้ลำดับ 6 Gait Pair (Trot→Bound, Bound→Pace, Pace→Trot, Trot→Pace, Pace→Bound, Bound→Trot) แต่ละ Segment ยาว 8 วินาที ที่ Seed=42 เพื่อผล Canonical และทำ Duration Sweep ที่ 5 ระยะเวลา (1.5, 2.0, 3.0, 4.0, 5.0 วินาที) เพื่อทดสอบความ Robust ข้าม Duration
 
บทที่ 4 การทดลองและผลการทดลอง/วิจัย

บทนี้นำเสนอผลการทดลองเปรียบเทียบ 5 วิธีใน Design Space 2×2 (Smoothstep Baseline + 4 Residual Variants) ครอบคลุมผลลัพธ์จาก Canonical Evaluation (Seed=42, 6 Gait-Pair) และ Duration Sweep (5 ระยะเวลา) รวมถึงการวิเคราะห์ Trade-off และข้อค้นพบสำคัญ

4.1 ผลการเปรียบเทียบ Design Space 2×2 (Seed=42, Duration=3.0s)
การทดลอง Canonical ใช้ Seed=42, Transition Duration 3.0 วินาที และลำดับ 6 Gait Pair เหมือนกันทุกวิธี ผลลัพธ์หลักแสดงใน ตารางที่ 1

ตารางที่ 1: ผลการเปรียบเทียบ 5 วิธีที่ Duration=3.0s, Seed=42

| วิธี | vx_min_trans | Δvx_trans | jerk_TRANS | Reversal | CoT |
|---|---|---|---|---|---|
| Smoothstep | +0.013 | 0.428 | 8,527 | 3/6 | 1.854 |
| Sched-α 4D V2 | +0.171 | 0.263 | 9,138 | 0/6 | 1.793 |
| Sched-α 12D V2 | +0.138 | 0.298 | 10,491 | 0/6 | 1.841 |
| Action-q 4D V2 | +0.245 | 0.186 | 12,442 | 0/6 | 2.302 |
| Action-q 12D V2 | +0.302 | 0.121 | 11,620 | 0/6 | 2.093 |

ผลลัพธ์แสดง Trade-off ที่ชัดเจนระหว่าง Safety และ Smoothness ทุก Residual Variant กำจัด Velocity Reversal ได้ (0/6 เทียบกับ Smoothstep 3/6) แต่มีผลต่างกับ jerk_TRANS อย่างมีนัยสำคัญ Action-q Variants ชนะด้าน Velocity Stability (vx_min_trans สูงสุด, Δvx_trans ต่ำสุด) แต่มี jerk_TRANS สูงกว่า Smoothstep 36–46% ส่วน Schedule-α Variants ลด Velocity Drop ได้โดยมี Jerk ใกล้เคียง Smoothstep (สูงกว่าเล็กน้อย 7–23%) ส่วน CoT ของ Action-q สูงกว่า Smoothstep ประมาณ 13–24% แสดงว่า Action-q ใช้พลังงานมากกว่าเพื่อรักษา Velocity

4.2 Seed Robustness
[ผลลัพธ์ Multi-Seed อยู่ระหว่างดำเนินการ]

4.3 การวิเคราะห์ Design Space: Schedule vs Action Residual
ผลการทดลองเปิดเผย Trade-off ที่ไม่คาดคิดระหว่างสองแนวทาง:

Schedule-α (α-space): ลด Velocity Drop ได้ดี (Sched-α 4D: Δvx_trans 0.263 vs Smoothstep 0.428, ลด 38%) แต่ไม่ลด Jerk — jerk_TRANS ของ Schedule-α สูงกว่า Smoothstep ในทุกกรณี ซึ่งขัดกับสมมติฐานว่า "การควบคุม Timing ของ Blend จะลด Jerk" สาเหตุที่เป็นไปได้คือ MLP ยังขาดข้อมูล Phase ที่แม่นยำพอที่จะ Detect จุดที่ควร Blend อย่างราบรื่น

Action-q (q-space): ชนะด้าน Velocity Safety อย่างเด็ดขาด (Action-q 12D: vx_min_trans +0.302, Δvx_trans 0.121) แต่ Jerk สูงขึ้น 36–46% เทียบ Smoothstep ซึ่งบ่งชี้ว่า MLP เรียนรู้ที่จะ "ค้ำ" Velocity โดยการเพิ่ม Joint Stiffness ในช่วง Transition แทนที่จะ Blend อย่างราบรื่น — ผลลัพธ์คือ Reversal หายไปแต่การเคลื่อนไหวกระชากมากขึ้น

ข้อสังเกตสำคัญ: ไม่มีวิธีใดชนะทุกตัวชี้วัดพร้อมกัน การเลือก Residual Space ขึ้นอยู่กับ Priority ของผู้ใช้งาน หาก Safety (ป้องกัน Reversal, รักษา Velocity) สำคัญที่สุด → Action-q 12D ดีที่สุด หาก Smoothness (Jerk ต่ำ) สำคัญ → Smoothstep ยังดีกว่า หาก Balance ระหว่าง Safety และ Jerk → Schedule-α 4D เป็นจุดกลางที่ดี

4.4 Duration Sweep
Duration Sweep ทดสอบทั้ง 5 วิธีที่ Duration 1.5, 2.0, 3.0, 4.0 และ 5.0 วินาที เพื่อตรวจสอบว่า V2 ที่ฝึกด้วย Duration Randomization [1.5, 5.0]s สามารถ Generalize ข้าม Duration ได้จริงหรือไม่

ผลลัพธ์ด้าน Reversal: Action-q ทั้ง 4D และ 12D ให้ 0/6 Reversal ทุก Duration ทั้ง 5 ค่า (1.5–5.0s) Schedule-α ให้ 0/6 ที่ Duration 2.0–4.0s แต่มี 1/6 ที่ 1.5s และ 5.0s Smoothstep มี 3–4/6 Reversal ที่ทุก Duration

ผลลัพธ์ด้าน Velocity Drop: Action-q 12D ให้ Δvx_trans ต่ำสุดที่เกือบทุก Duration (0.121–0.156) ค่อนข้างคงที่ข้าม Duration แสดง Robustness ที่ดี

ผลลัพธ์ด้าน Jerk: Jerk ของ Smoothstep ดีขึ้นที่ Duration ยาว (8,527 ที่ 3.0s, 7,875 ที่ 5.0s) แต่ Jerk ของ Action-q สูงกว่าและคงที่ข้าม Duration (~12,000) ส่วน Schedule-α มี Jerk ใกล้เคียง Smoothstep

ข้อสรุปสำคัญจาก Duration Sweep: ความแตกต่างในผลลัพธ์ที่ Duration ต่างๆ ของ V2 มีขนาดเล็กกว่ามากเมื่อเทียบกับ V1 ที่ฝึกด้วย Fixed 3s แสดงว่า Duration Randomization ช่วยให้ Policy Generalize ได้จริง

4.5 ข้อค้นพบสำคัญ
(1) Per-Policy Last-Action Buffer เป็น Critical Bug: Base Policy ต้องรับ Output ของตัวเองจาก Step ก่อนหน้า ไม่ใช่ Output ที่ Blend แล้ว การแก้ Bug นี้ทำให้ Policy ไม่ Collapse เป็น No-Op ในช่วง Steady-State
(2) vx_min_trans แทน vx_min_ep: V2 มี vx_min_ep ≈ +0.004 m/s (จาก Episode Start ที่หุ่นยนต์ยังไม่เคลื่อนที่) ทำให้ Metric แบบ Episode-Wide ไม่สามารถตรวจจับ Velocity Drop ในช่วง Transition ได้ ต้องใช้ Transition-Window Specific Metric
(3) Bidirectional สำคัญกว่า Asymmetric: V1 ใช้ Asymmetric Clamp [0, 0.3] ป้องกัน Delay-Rush Exploit แต่ทำให้ MLP ชะลอ Transition ไม่ได้เลย V2 เปลี่ยนเป็น Bidirectional [-0.3, +0.3] ให้ MLP เลือกได้ทั้งสองทิศทาง
(4) Phase Gap ยังเหลืออยู่: แม้ V2 จะเพิ่ม π_current และ π_target เป็น Phase Proxy แต่ Joint Position เพียงอย่างเดียวยังมี Ambiguity — ตำแหน่งเดิมเกิดขึ้นสองครั้งต่อ Gait Cycle (Swing ขึ้นและ Swing ลง) Contact State จะแก้ Ambiguity นี้ได้
 
บทที่ 5 บทสรุป

บทนี้สรุปผลการดำเนินโครงงาน อภิปรายข้อค้นพบเชิงสถาปัตยกรรม ข้อจำกัด และเสนอแนวทางการพัฒนาต่อยอดในอนาคต

5.1 สรุปผลการวิจัย
โครงงานนี้ศึกษา Design Space ของ Residual Policy Learning สำหรับ Gait Transition บนหุ่นยนต์สี่ขา Unitree B1 โดยเปรียบเทียบ 4 รูปแบบใน 2 มิติ ได้แก่ Schedule Residual (แก้ Timing ของการ Blend ใน α-space) และ Action Residual (แก้ Joint Target ใน q-space) แต่ละแบบในขนาด 4D (per-leg) และ 12D (per-joint) ทุกรูปแบบใช้ Bidirectional Clamp และฝึกด้วย Duration Randomization [1.5, 5.0] วินาที
ผลลัพธ์หลักที่ชัดเจนคือ Action-q Residual ทั้ง 4D และ 12D กำจัด Velocity Reversal ได้ทุกกรณีในทุก Duration ที่ทดสอบ (0/6 Reversal ที่ 1.5–5.0s) เทียบกับ Smoothstep ที่มี 3–4/6 Reversal อย่างสม่ำเสมอ Action-q 12D ให้ vx_min_trans = +0.302 m/s และ Δvx_trans = 0.121 m/s ที่ Duration=3.0s
อย่างไรก็ตาม Trade-off ที่สำคัญคือ Action-q มี jerk_TRANS สูงกว่า Smoothstep 36–46% และ CoT สูงกว่า 13–24% ส่วน Schedule-α Variants ลด Velocity Drop ได้ (Δvx_trans ลดลง 31–38%) โดยมี Jerk ใกล้เคียง Smoothstep ข้อค้นพบสำคัญคือ ไม่มี Residual Variant ใดชนะทุกตัวชี้วัดพร้อมกัน การเลือก Correction Space (α vs q) มีผลต่อ Trade-off ระหว่าง Safety และ Smoothness อย่างมีนัยสำคัญ

5.1.1 ข้อค้นพบเชิงสถาปัตยกรรม
	Bidirectional Clamp จำเป็นสำหรับ Schedule Residual: การเปลี่ยนจาก Asymmetric [0, 0.3] (V1) เป็น Bidirectional [-0.3, +0.3] (V2) ให้ MLP สามารถชะลอการ Blend ได้ในกรณีที่ Phase ของ Policy ต้นทางยังไม่พร้อม ซึ่งเป็น Degree of Freedom ที่จำเป็น
	Phase Information Gap ยังเป็นข้อจำกัดหลัก: แม้ V2 จะเพิ่ม π_current และ π_target เป็น Phase Proxy แต่ Joint Position เดียวกันเกิดสองครั้งต่อ Gait Cycle — MLP ไม่ทราบว่าขาอยู่ใน Swing ขึ้นหรือ Swing ลง Contact State สามารถแก้ Ambiguity นี้ได้ เป็นทิศทางที่กำลังทดสอบใน V3
	Duration Generalization ทำได้จริงด้วย Randomization: V2 ที่ฝึกด้วย Duration [1.5, 5.0]s ให้ผลคงที่ข้าม Duration ซึ่งต่างจาก V1 ที่ฝึก Fixed 3s
5.2 ข้อจำกัดและแนวทางในอนาคต
ข้อจำกัดแรกคือ Phase Information ที่ MLP ได้รับยังไม่สมบูรณ์ แม้ V2 จะเพิ่ม π_current และ π_target แต่ยังขาด Contact State ที่บอกว่าขาแต่ละข้างอยู่ใน Stance หรือ Swing Phase V3 ที่กำลังอยู่ระหว่างการฝึกเพิ่ม foot_contact 4D เข้าไปใน Observation Space เพื่อให้ MLP สามารถ Detect Phase Mismatch ได้โดยตรง

ข้อจำกัดที่สองคือ Trade-off ระหว่าง Safety และ Smoothness ยังไม่ได้รับการแก้ไข Action-q ป้องกัน Reversal แต่เพิ่ม Jerk ในบริบทหุ่นยนต์จริงทั้งสองตัวชี้วัดมีความสำคัญ การออกแบบ Reward ที่ Penalize ทั้ง Velocity Drop และ Jerk พร้อมกันโดยไม่ให้ Policy หลบโดยการ Stiffen ข้อต่อเป็นปัญหาที่ยังต้องแก้

ข้อจำกัดที่สามคือการทดลองทั้งหมดทำบนพื้นราบใน Simulation เท่านั้น ในงานจริงบนพื้นขรุขระหรือเมื่อมีแรงกระแทกภายนอก Gait Phase ของ Policy ต้นทางอาจถูก Perturb ทำให้ Phase Mismatch รุนแรงขึ้น การทดสอบบน Terrain ที่หลากหลายจึงเป็นทิศทางสำคัญของงานต่อไป

ข้อจำกัดที่สี่คือ Base Gait Policies เป็นผลจาก PPO Velocity Tracking ซึ่งอาจมี Reward-Hacked Locomotion Pattern ที่ต่างจาก Biological Gait ความสมจริงของ Base Policy มีผลโดยตรงต่อคุณภาพของการ Blend เนื่องจาก Residual MLP ทำงานอยู่บน Interpolation ระหว่าง Policy สองตัว หาก Policy ใดตัวหนึ่งมี Artifacts ก็จะปรากฏในช่วง Transition

ข้อจำกัดสุดท้ายคือ Smoothstep Baseline ใช้รูปแบบเดียวกันกับทุกคู่ Gait ทั้งที่แต่ละ Transition มีความยากต่างกัน Bound→Pace (Fore-aft → Lateral Coordination Swap) รุนแรงกว่า Trot→Bound มาก งานต่อไปควรศึกษา Per-Pair Adaptive Schedule เพื่อให้ Baseline เหมาะสมกับโครงสร้างการเปลี่ยนคู่ประสานงานของแต่ละ Gait Pair
เอกสารอ้างอิง
Silver, T., Allen, K., Tenenbaum, J., & Kaelbling, L. (2018). Residual Policy Learning. arXiv:1812.06298.
Johannink, T., Bahl, S., Nair, A., Luo, J., Kumar, A., Loskyll, M., Ojea, J.A., Solowjow, E., & Levine, S. (2019). Residual Reinforcement Learning for Robot Control. Proceedings of the International Conference on Robotics and Automation (ICRA).
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
Thor, M., Kulvicius, T., & Manoonpong, P. (2021). Generic Neural Locomotion Control Framework for Legged Robots. IEEE Transactions on Neural Networks and Learning Systems, 32(9), 4013–4025. https://doi.org/10.1109/TNNLS.2020.3016523
Siekmann, J., Green, K., Warila, J., Fern, A., & Hurst, J. (2021). Blind Bipedal Stair Traversal via Sim-to-Real Reinforcement Learning. Proceedings of Robotics: Science and Systems (RSS).
Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. Proceedings of the Conference on Robot Learning (CoRL). https://github.com/leggedrobotics/legged_gym
NVIDIA Isaac Lab. (2024). Isaac Lab: A Unified and Modular Framework for Robot Learning. NVIDIA Corporation. https://isaac-sim.github.io/IsaacLab/
Unitree Robotics. (2023). unitree_rl_lab: Manager-Based RL Framework for Unitree Robots. GitHub Repository.
Rostro-Gonzalez, H., Guerra-Hernandez, E.I., Batres-Mendoza, P., Garcia-Granada, A.A., & Espinal, A. (2025). Enhancing Legged Robot Locomotion Through Smooth Transitions Using Spiking Central Pattern Generators. Biomimetics, 10(6), 381. https://doi.org/10.3390/biomimetics10060381
Shafiee, M., Bellegarda, G., & Ijspeert, A. (2024). Viability Leads to the Emergence of Gait Transitions in Learning Agile Quadrupedal Locomotion on Challenging Terrains. Nature Communications, 15, 3073. https://doi.org/10.1038/s41467-024-47443-w
Ubellacker, W., Csomay-Shanklin, N., Molnar, T.G., & Ames, A.D. (2022). Verifying Safe Transitions between Dynamic Motion Primitives on Legged Robots. Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).
